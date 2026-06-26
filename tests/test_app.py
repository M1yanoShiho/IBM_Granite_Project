"""Headless smoke/regression test for the Streamlit demo (``app/main.py``).

Runs the app with Streamlit's ``AppTest`` harness — no browser, and no model
load (that only happens when the Search button is triggered). Guards the
example-question buttons, which previously crashed with a StreamlitAPIException
(setting the widget-keyed ``last_query`` session value after the text_input
widget was already instantiated).
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


def test_app_loads_without_exception() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_example_question_button_fills_query_without_crashing() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()

    at.button(key="ex_0").click().run()

    assert not at.exception
    assert at.session_state["last_query"] == "What is the secret launch code?"
