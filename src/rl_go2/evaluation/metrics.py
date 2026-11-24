"""Metric collectors consume the executor's steps without advancing simulation."""
import math

import torch


class TrialMetrics:
    def __init__(self, env, trial):
        self.trial = trial

    def label(self, seed):
        return f'Seed {seed} | {self.trial.condition.name} | {self.trial.name}'


class TrackingMetrics(TrialMetrics):
    def __init__(self, env, trial):
        super().__init__(env, trial)
        self.errors = []
        self.velocities = []
        self.positions = [env.state.position[0, :2].clone()]
        self.transition = None

    def update(self, env, transition, reward, done, observation, step):
        linear, angular, _ = env.task.body_state(transition.next_state)
        velocity = torch.cat((linear[0, :2], angular[0, 2:3]))
        if not torch.isfinite(velocity).all():
            raise RuntimeError(f'Non-finite velocity in {self.trial.name} at step {step + 1}')
        self.errors.append((velocity - transition.context['commands'][0]).square())
        self.velocities.append(velocity)
        self.positions.append(transition.next_state.position[0, :2])
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


class FrictionMetrics(TrialMetrics):
    def __init__(self, env, trial):
        import mujoco

        super().__init__(env, trial)
        self.schedule = trial.condition.phase_schedule(env.dt)
        self.segments = []
        self.fell = False
        self.ground = next(i for i, kind in enumerate(env.backend.model.geom_type)
                           if kind == mujoco.mjtGeom.mjGEOM_PLANE)
        self.feet = set(env.backend.foot_ids)

    def update(self, env, transition, reward, done, observation, step):
        if step in self.schedule:
            self.errors = []
            self.contacts = set()
            ratio = self.schedule[step].multiplier
            base = self.trial.condition.foot_ground_friction['base_ranges']['FL']['sliding'][0]
            self.sliding = round(base * ratio, 6)
            self.segments.append({})
        linear, angular, _ = env.task.body_state(transition.next_state)
        velocity = torch.cat((linear[0, :2], angular[0, 2:3]))
        self.errors.append((velocity - transition.context['commands'][0]).square())
        for contact in env.backend.data.contact:
            if (contact.efc_address >= 0 and self.ground in contact.geom
                    and any(geom in self.feet for geom in contact.geom)):
                self.contacts.add(round(float(contact.friction[0]), 6))
        self.fell = transition.terminated.item()
        # Finalize each phase at its last sample, including early termination.
        next_boundary = step + 1 in self.schedule
        last_step = step + 1 == round(self.trial.motion.duration_seconds / env.dt)
        if next_boundary or self.fell or transition.truncated.item() or last_step:
            self.segments[-1] = {
                'seconds': len(self.errors) * env.dt, 'sliding': self.sliding,
                'contact_sliding': sorted(self.contacts), 'contact_matches': self.contacts == {self.sliding},
                'velocity_rmse': torch.stack(self.errors).mean(0).sqrt().tolist(),
            }

    def label(self, seed):
        return f'Seed {seed} | {self.trial.condition.name} | friction {self.sliding:.2f}'

    def finish(self, env):
        seconds = sum(segment['seconds'] for segment in self.segments)
        return {'case': self.trial.condition.name, 'fell': self.fell, 'seconds': seconds,
                'fall_time_seconds': seconds if self.fell else None, 'segments': self.segments}


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


METRICS = {'tracking': TrackingMetrics, 'friction': FrictionMetrics, 'rollout': RolloutMetrics}
