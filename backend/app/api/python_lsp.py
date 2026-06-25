"""
Jedi-powered Python completion endpoint for the in-browser code editor.

A small stub (show_image, show_text, config) is prepended to the user's code
before passing to Jedi so those builtins appear in autocomplete results.
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/python", tags=["python"])

# Stubs injected before user code so Jedi knows about CV-FLOW builtins
_STUB = (
    "def show_image(img, label: str = '') -> None: ...\n"
    "def show_text(text) -> None: ...\n"
    "config: dict = {}\n"
)
_STUB_LINES = 3  # number of lines in _STUB (no trailing blank)


class CompleteRequest(BaseModel):
    code:   str
    line:   int   # 1-based line in user code
    column: int   # 0-based column


@router.post("/complete")
async def complete(req: CompleteRequest):
    try:
        import jedi  # lazy import — not required at startup
        combined  = _STUB + req.code
        jedi_line = req.line + _STUB_LINES
        script    = jedi.Script(combined)
        results   = script.complete(jedi_line, req.column)
        return {
            "completions": [
                {
                    "name":        c.name,
                    "type":        c.type,
                    "description": getattr(c, "description", ""),
                }
                for c in results[:100]
            ]
        }
    except ImportError:
        return {"completions": [], "error": "jedi not installed — pip install jedi"}
    except Exception as exc:
        return {"completions": [], "error": str(exc)}
