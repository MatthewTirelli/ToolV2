"""Agent pipeline modules."""

__all__ = ["run_measles_multi_agent_pipeline"]


def run_measles_multi_agent_pipeline(*args, **kwargs):
    from agents.report_writer_agent import run_measles_multi_agent_pipeline as _impl
    return _impl(*args, **kwargs)
