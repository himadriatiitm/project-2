import sys
import traceback


class InterpreterError(Exception):
    pass


default_exec = exec


def exec(cmd, globals=None, locals=None):
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
        return

    # a little bit of lookahead
    cmdlines = cmd.splitlines()
    try:
        lookbehind = "\n".join(cmdlines[line_number - 2 : line_number])
    except IndexError:
        lookbehind = ""
    line_itself = cmdlines[line_number]
    try:
        lookahead = "\n".join(cmdlines[line_number + 1 : line_number + 3])
    except IndexError:
        lookahead = ""
    raise InterpreterError(
        f"""
{lookbehind}
...
{error_class} at line {line_number} of main.py:
{line_itself}
^-- {detail}
...
{lookahead}
"""
    )
