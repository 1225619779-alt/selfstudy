import argparse
import re
from pathlib import Path


def _replace_assignment(text: str, var_name: str, new_value: str) -> tuple[str, int]:
    pattern = rf'(?m)^(\s*{re.escape(var_name)}\s*=\s*)([^\n#]+?)(\s*(#.*)?)$'
    repl = rf'\g<1>{new_value}\g<3>'
    return re.subn(pattern, repl, text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a copy of evaluation_event_trigger.py with patched tau_verify and total_run.')
    parser.add_argument('--src', default='evaluation_event_trigger.py')
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--total_run', type=int, default=50)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.out)
    text = src.read_text(encoding='utf-8')

    text, n_total = _replace_assignment(text, 'total_run', str(args.total_run))
    if n_total == 0:
        raise RuntimeError('Failed to replace total_run. Please inspect evaluation_event_trigger.py manually.')

    text, n_tau = _replace_assignment(text, 'tau_verify', str(args.tau))
    if n_tau == 0:
        raise RuntimeError('Failed to replace tau_verify. Please inspect evaluation_event_trigger.py manually.')

    dst.write_text(text, encoding='utf-8')
    print('Generated:', dst)
    print('total_run =', args.total_run)
    print('tau_verify =', args.tau)


if __name__ == '__main__':
    main()
