"""Metric collectors consume the executor's steps without advancing simulation."""
import math

import torch


RECOVERY_CRITERIA = {
    'planar_velocity_error_m_s': 0.1,
    'yaw_rate_error_rad_s': 0.15,
    'tilt_deg': 15.0,
    'hold_seconds': 0.5,
}


class PushRecoveryMetrics:
    """Track disturbances and the first sustained return to commanded motion."""

    def __init__(self, event, dt):
        self.event = event
        self.dt = dt
        self.hold_steps = math.ceil(RECOVERY_CRITERIA['hold_seconds'] / dt)
        self.hold = 0
        self.recovery_seconds = None
        self.peak_tilt_deg = 0.0
        self.peak_planar_error = 0.0

    def update(self, env, transition, step) -> None:
        if step < self.event.start_step:
            return
        state = transition.next_state
        linear, angular, gravity = env.task.body_state(state)
        command = transition.context['commands'][0]
        planar_error = (linear[0, :2] - command[:2]).norm().item()
        yaw_error = abs(angular[0, 2].item() - command[2].item())
        tilt = math.degrees(math.acos((-gravity[0, 2]).clamp(-1, 1).item()))
        self.peak_tilt_deg = max(self.peak_tilt_deg, tilt)
        self.peak_planar_error = max(self.peak_planar_error, planar_error)
        if step < self.event.end_step or self.recovery_seconds is not None:
            return
        stable = (
            planar_error < RECOVERY_CRITERIA['planar_velocity_error_m_s']
            and yaw_error < RECOVERY_CRITERIA['yaw_rate_error_rad_s']
            and tilt < RECOVERY_CRITERIA['tilt_deg']
            and not transition.terminated[0].item()
        )
        self.hold = self.hold + 1 if stable else 0
        if self.hold >= self.hold_steps:
            self.recovery_seconds = (step + 1 - self.event.end_step) * self.dt

    def finish(self, steps: int, fell: bool) -> dict:
        if steps <= self.event.start_step:
            return {'status': 'fell_before_push', 'recovery_seconds': None}
        if fell:
            status = 'fell_after_push_onset'
        elif self.recovery_seconds is None:
            status = 'not_recovered'
        else:
            status = 'recovered'
        return {
            'status': status,
            'recovery_seconds': self.recovery_seconds,
            'peak_tilt_deg': self.peak_tilt_deg,
            'peak_planar_velocity_error_m_s': self.peak_planar_error,
        }


class TrialMetrics:
    def __init__(self, env, trial):
        self.trial = trial

    def label(self, seed):
        if self.trial.scene is not None:
            return f'Seed {seed} | {self.trial.scene} | {self.trial.condition.name} | {self.trial.name}'
        return f'Seed {seed} | {self.trial.condition.name} | {self.trial.name}'


class TrackingMetrics(TrialMetrics):
    def __init__(self, env, trial):
        super().__init__(env, trial)
        self.errors = []
        self.velocities = []
        self.positions = [env.state.position[0, :2].clone()]
        self.stand_heights = [] if trial.name == 'stand' else None
        self.transition = None

    def update(self, env, transition, reward, done, observation, step):
        linear, angular, _ = env.task.body_state(transition.next_state)
        velocity = torch.cat((linear[0, :2], angular[0, 2:3]))
        if not torch.isfinite(velocity).all():
            raise RuntimeError(f'Non-finite velocity in {self.trial.name} at step {step + 1}')
        self.errors.append((velocity - transition.context['commands'][0]).square())
        self.velocities.append(velocity)
        self.positions.append(transition.next_state.position[0, :2])
        if self.stand_heights is not None:
            self.stand_heights.append(transition.next_state.position[0, 2])
        self.transition = transition

    def finish(self, env):
        seconds = len(self.errors) * env.dt
        fell = self.transition.terminated.item()
        errors = torch.stack(self.errors)
        transient_steps = round(2 / env.dt)
        result = {
            'group': self.trial.group, 'case': self.trial.name,
            'segments': [{'start_seconds': step * env.dt, 'command': list(command)}
                         for step, command in self.trial.motion.command_schedule(env.dt).items()],
            'steps': len(self.errors), 'seconds': seconds, 'fell': fell,
            'fall_time_seconds': seconds if fell else None,
            'velocity_rmse': errors.mean(0).sqrt().tolist(),
            'velocity_rmse_after_2s': errors[transient_steps:].mean(0).sqrt().tolist()
            if len(errors) > transient_steps else None,
        }
        if self.trial.motion.stop_at_seconds is not None:
            result.update(self.stop_metrics(env.dt, not fell))
        if self.stand_heights is not None:
            result.update(
                min_base_height_m=torch.stack(self.stand_heights).min().item(),
                final_planar_displacement_m=(self.positions[-1] - self.positions[0]).norm().item(),
            )
        if env.foot_friction is not None:
            result['foot_ground_friction'] = self.transition.context['foot_ground_friction'][0].reshape(4, 3).tolist()
        if env.base_mass_kg is not None:
            result['base_mass_kg'] = env.base_mass_kg.item()
        if env.base_com_offset_m is not None:
            result['base_com_offset_m'] = env.base_com_offset_m[0].tolist()
        return result

    def stop_metrics(self, dt, completed):
        switch_step = round(self.trial.motion.stop_at_seconds / dt)
        hold_steps = math.ceil(0.5 / dt)
        tail_steps = round(2 / dt)
        hold = 0
        settled = None
        for index, velocity in enumerate(self.velocities[switch_step:]):
            slow = velocity[:2].norm().item() < 0.1 and abs(velocity[2].item()) < 0.15
            hold = hold + 1 if slow else 0
            if hold >= hold_steps:
                settled = (index + 1) * dt
                break
        return {
            'time_to_settle_seconds': settled,
            'final_2s_displacement_m': (self.positions[-1] - self.positions[-1 - tail_steps]).norm().item()
            if completed and tail_steps > 0 and len(self.positions) > tail_steps else None,
        }


TERRAIN_SUCCESS_CRITERIA = {
    'max_forward_velocity_mse_after_2s_m2_s2': 0.04,
}


class TerrainMetrics(TrackingMetrics):
    """Judge terrain tracking and report motion and posture diagnostics."""

    def __init__(self, env, trial):
        super().__init__(env, trial)
        self.quaternions = []

    def update(self, env, transition, reward, done, observation, step):
        super().update(env, transition, reward, done, observation, step)
        self.quaternions.append(transition.next_state.quaternion[0])

    def finish(self, env):
        result = super().finish(env)
        rmse = result['velocity_rmse_after_2s']
        mse = rmse[0] ** 2 if rmse is not None else None
        final_x = self.positions[-1][0].item()
        reasons = []
        if result['fell']:
            reasons.append('fall')
        if mse is None:
            reasons.append('insufficient_tracking_samples')
        elif mse > TERRAIN_SUCCESS_CRITERIA['max_forward_velocity_mse_after_2s_m2_s2']:
            reasons.append('tracking_mse')
        roll_pitch = torch.stack(self.quaternions)[:, 1:3]
        upright = (1 - 2 * roll_pitch.square().sum(dim=1)).clamp(-1, 1)
        result.update(
            success=not reasons, failure_reasons=reasons,
            forward_velocity_mse_after_2s_m2_s2=mse,
            final_x_m=final_x,
            forward_displacement_m=final_x - self.positions[0][0].item(),
            peak_tilt_deg=math.degrees(math.acos(upright.min().item())),
        )
        return result


class FrictionMetrics(TrackingMetrics):
    def __init__(self, env, trial):
        import mujoco

        super().__init__(env, trial)
        self.schedule = trial.condition.phase_schedule(env.dt)
        self.segments = []
        self.ground = next(i for i, kind in enumerate(env.backend.model.geom_type)
                           if kind == mujoco.mjtGeom.mjGEOM_PLANE)
        self.feet = set(env.backend.foot_ids)

    def update(self, env, transition, reward, done, observation, step):
        super().update(env, transition, reward, done, observation, step)
        if step in self.schedule:
            self.phase_start = step
            self.contacts = set()
            ratio = self.schedule[step].multiplier
            base = self.trial.condition.foot_ground_friction['base_ranges']['FL']['sliding'][0]
            self.sliding = round(base * ratio, 6)
            self.segments.append({})
        # The simulator has already reset on a terminal step.
        if not done[0].item():
            for contact in env.backend.data.contact:
                if (contact.efc_address >= 0 and self.ground in contact.geom
                        and any(geom in self.feet for geom in contact.geom)):
                    self.contacts.add(round(float(contact.friction[0]), 6))
        # Finalize each phase at its last sample, including early termination.
        next_boundary = step + 1 in self.schedule
        last_step = step + 1 == round(self.trial.motion.duration_seconds / env.dt)
        if next_boundary or done[0].item() or last_step:
            phase_errors = self.errors[self.phase_start:]
            self.segments[-1] = {
                'seconds': len(phase_errors) * env.dt, 'sliding': self.sliding,
                'contact_sliding': sorted(self.contacts), 'contact_matches': self.contacts == {self.sliding},
                'velocity_rmse': torch.stack(phase_errors).mean(0).sqrt().tolist(),
            }

    def label(self, seed):
        return f'{super().label(seed)} | friction {self.sliding:.2f}'

    def finish(self, env):
        result = super().finish(env)
        result['command_segments'] = result['segments']
        result['segments'] = self.segments
        return result


class RolloutMetrics(TrialMetrics):
    def __init__(self, env, trial):
        super().__init__(env, trial)
        self.total_reward = 0.0
        self.resets = 0
        self.steps = 0

    def update(self, env, transition, reward, done, observation, step):
        self.total_reward += reward.mean().item()
        self.resets += int(done.sum())
        self.steps += 1
        self.observation_shape = list(observation.shape)

    def finish(self, env):
        return {'steps': self.steps, 'seconds': self.steps * env.dt,
                'observation_shape': self.observation_shape,
                'mean_total_reward': self.total_reward, 'episode_resets': self.resets, 'policy_updates': 0}


METRICS = {
    'tracking': TrackingMetrics, 'terrain': TerrainMetrics,
    'friction': FrictionMetrics, 'rollout': RolloutMetrics,
}
