import sys
import traceback
import contextlib
import io

default_exec = exec

def exec(cmd, globals=None, locals=None):
    out = io.StringIO()
    with (
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(out),
    ):
        try:
            default_exec(cmd, globals, locals)
        except SyntaxError as err:
            error_class = err.__class__.__name__
            detail = err.args[0]
            line_number = err.lineno
        except Exception as err:
            error_class = err.__class__.__name__
            detail = err.args[0]
            cl, exc, tb = sys.exc_info()
            line_number = traceback.extract_tb(tb)[1][1] - 1
        else:
            stdout = out.getvalue()
            return stdout

        # a little bit of lookahead
        cmdlines = cmd.splitlines()
        stdout = out.getvalue()
        try:
            lookbehind = "\n".join(cmdlines[line_number - 2 : line_number])
        except IndexError:
            lookbehind = ""
        line_itself = cmdlines[line_number]
        try:
            lookahead = "\n".join(cmdlines[line_number + 1 : line_number + 3])
        except IndexError:
            lookahead = ""
        return f"""
{stdout}

Traceback:
{lookbehind}
...
{error_class} at line {line_number} of main.py:
{line_itself}
^-- {detail}
...
{lookahead}
    """
