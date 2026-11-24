"""Read-only Go2 experiment dashboard for an SSH-forwarded browser."""

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

from dashboard.artifacts import (
    ArtifactError,
    Evaluation,
    child_directories,
    evaluation_cases,
    evaluations,
    local_file,
    read_json,
    run_directory,
    training_progress,
    trial_totals,
)

HERE = Path(__file__).resolve().parent
BEIJING = timezone(timedelta(hours=8))
templates = Jinja2Templates(
    env=Environment(loader=FileSystemLoader(HERE / 'templates'), autoescape=True),
)


def updated_at(path: Path) -> str:
    """Show directory modification time in the project's timezone."""
    return datetime.fromtimestamp(path.stat().st_mtime, BEIJING).strftime('%Y-%m-%d %H:%M')


def evaluation_label(evaluation: Evaluation) -> str:
    """Format the unambiguous date suffix used by current evaluation output."""
    match = re.search(r'-(\d{8})-(\d{6})(?:-\d{6})?$', evaluation.name)
    if match:
        try:
            timestamp = datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S')
        except ValueError:
            return updated_at(evaluation.path)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return updated_at(evaluation.path)


def describe_evaluation(item: Evaluation) -> dict:
    """Describe an evaluation for the run page even when one report is invalid."""
    try:
        summary = read_json(item.path / 'summary.json')
        config = read_json(item.path / 'eval-config.json') if summary is None else None
        error = None
    except ArtifactError as exc:
        summary, config, error = None, None, str(exc)
    report = summary or {}
    metadata = summary or config or {}
    complete = summary is not None or not (item.path / 'eval-config.json').exists()
    cases = evaluation_cases(item, complete=complete)
    trials, falls = trial_totals(report)
    return {
        'name': item.name, 'iteration': item.iteration, 'date': evaluation_label(item),
        'task': metadata.get('task') or re.split(r'-(?:mujoco|genesis)-', item.name, maxsplit=1)[0],
        'backend': metadata.get('backend'), 'machine': metadata.get('machine'),
        'trials': trials, 'falls': falls, 'videos': sum(case.video is not None for case in cases),
        'error': error, 'has_summary': summary is not None,
    }


def create_app(output_root: Path | None = None) -> FastAPI:
    """Build the read-only dashboard for an existing output directory."""
    root = (output_root or Path(os.environ.get('GO2_DASHBOARD_OUTPUT', HERE.parent / 'output'))).resolve()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount('/static', StaticFiles(directory=HERE / 'static'), name='static')

    def find_evaluation(run_name: str, iteration: int, evaluation_name: str) -> Evaluation:
        run = run_directory(root, run_name)
        for item in evaluations(run):
            if item.iteration == iteration and item.name == evaluation_name:
                return item
        raise ArtifactError('找不到该轮次下的评测目录。')

    def page(request: Request, template: str, status: int = 200, **context: object) -> HTMLResponse:
        return templates.TemplateResponse(
            request, template, context, status_code=status,
            headers={'Cache-Control': 'no-store'},
        )

    @app.get('/', response_class=HTMLResponse)
    def index(request: Request, q: str = '') -> HTMLResponse:
        if not root.is_dir():
            return page(request, 'error.jinja', 503, message='找不到输出目录，请检查 GO2_DASHBOARD_OUTPUT。')
        directories = sorted(
            child_directories(root), key=lambda path: path.stat().st_mtime, reverse=True,
        )
        runs = [
            {'name': run.name, 'updated': updated_at(run), 'evaluations': len(evaluations(run))}
            for run in directories if q.strip().casefold() in run.name.casefold()
        ]
        return page(request, 'index.jinja', runs=runs, q=q)

    @app.get('/runs/{run_name}', response_class=HTMLResponse)
    def run_detail(request: Request, run_name: str) -> HTMLResponse:
        try:
            run = run_directory(root, run_name)
            config = read_json(run / 'training' / 'config.json') or {}
        except ArtifactError as exc:
            return page(request, 'error.jinja', 404, message=str(exc))
        trainer = config.get('trainer')
        budget = trainer.get('iterations') if isinstance(trainer, dict) else None
        if type(budget) is not int or budget <= 0:
            budget = None
        metric_error = None
        try:
            latest, series = training_progress(run, budget)
        except ArtifactError as exc:
            latest, series, metric_error = {}, {}, str(exc)
        exit_path = local_file(run / 'exit-code')
        exit_code = exit_path.read_text(encoding='utf-8').strip() if exit_path else ''
        return page(
            request, 'run.jinja', run_name=run_name, latest=latest, series=series,
            budget=budget, exit_code=exit_code, metric_error=metric_error,
            items=[describe_evaluation(item) for item in evaluations(run)],
        )

    @app.get('/runs/{run_name}/eval/{iteration}/{evaluation_name}', response_class=HTMLResponse)
    def evaluation_detail(
        request: Request, run_name: str, iteration: int, evaluation_name: str, case: str = '',
    ) -> HTMLResponse:
        try:
            item = find_evaluation(run_name, iteration, evaluation_name)
            summary = read_json(item.path / 'summary.json')
            config = read_json(item.path / 'eval-config.json')
            cases = evaluation_cases(item, complete=summary is not None or config is None)
            if case and not any(entry.name == case for entry in cases):
                raise ArtifactError('找不到该场景或工况的报告及回放。')
        except ArtifactError as exc:
            return page(request, 'error.jinja', 404, run_name=run_name, message=str(exc))

        selected = next((entry for entry in cases if entry.name == case), None)
        if selected is None and cases:
            selected = next((entry for entry in cases if entry.video), cases[0])
        report_error = None
        report = None
        if selected and selected.assessment:
            try:
                report = read_json(selected.assessment)
            except ArtifactError as exc:
                report_error = str(exc)
        metadata = {**(config or {}), **(summary or {})}
        return page(
            request, 'evaluation.jinja', run_name=run_name, item=item, cases=cases,
            selected=selected, report=report or {}, summary=summary or {},
            report_error=report_error, has_summary=summary is not None,
            has_config=config is not None, metadata=metadata, date=evaluation_label(item),
            totals=trial_totals(report or summary or {}),
        )

    @app.api_route(
        '/runs/{run_name}/eval/{iteration}/{evaluation_name}/artifact/{kind}',
        methods=['GET', 'HEAD'],
    )
    def artifact(
        run_name: str, iteration: int, evaluation_name: str, kind: str, case: str = '',
    ) -> Response:
        try:
            item = find_evaluation(run_name, iteration, evaluation_name)
        except ArtifactError as exc:
            raise HTTPException(404, str(exc)) from exc
        reports = {'summary': 'summary.json', 'config': 'eval-config.json'}
        if kind in reports:
            path = local_file(item.path / reports[kind])
        elif kind in {'assessment', 'video', 'preview'}:
            complete = local_file(item.path / 'summary.json') is not None
            complete = complete or local_file(item.path / 'eval-config.json') is None
            entry = next(
                (entry for entry in evaluation_cases(item, complete=complete) if entry.name == case),
                None,
            )
            path = {'assessment': entry.assessment, 'video': entry.video, 'preview': entry.preview}[kind] if entry else None
        else:
            path = None
        if path is None:
            raise HTTPException(404, '找不到该产物。')
        return FileResponse(path, headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})

    return app
