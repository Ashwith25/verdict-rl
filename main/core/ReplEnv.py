class PersistentREPL:
    def __init__(self, globals):
        # Create a dict as the execution globals
        self.globals = globals

    def execute(self, code: str) -> str:
        # execute code and capture stdout (truncate)
        import io, contextlib

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, self.globals, self.globals)
        except Exception as e:
            print(f"[RUNTIME ERROR] {e}", file=buf)
        out = buf.getvalue()
        # hard truncate stdout
        max_len = 5000
        if len(out) > max_len:
            out = out[:max_len] + "\n...[truncated]..."
        return out