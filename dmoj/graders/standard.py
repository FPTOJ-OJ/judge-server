import glob
import itertools
import logging
import os
import subprocess
from typing import Optional, cast
from dmoj.checkers import CheckerOutput
from dmoj.cptbox import TracedPopen
from dmoj.cptbox.lazy_bytes import LazyBytes
from dmoj.error import OutputLimitExceeded
from dmoj.executors import executors
from dmoj.executors.base_executor import BaseExecutor
from dmoj.graders.base import BaseGrader
from dmoj.problem import TestCase
from dmoj.result import CheckerResult, Result

log = logging.getLogger('dmoj.graders')

class StandardGrader(BaseGrader):
    def grade(self, case: TestCase) -> Result:
        result = Result(case)

        io_mode = case.config.get('io_mode', case.problem.config.get('io_mode', 'std'))
        input_file = case.config.get('input_file', case.problem.config.get('input_file'))
        output_file = case.config.get('output_file', case.problem.config.get('output_file'))

        if io_mode == 'file' and (not input_file or not output_file):
            log.error('File I/O mode specified but input_file or output_file missing')
            result.result_flag = Result.IE
            result.feedback = 'Invalid file I/O configuration'
            return result

        self.binary = self._generate_binary()

        input_path = None
        output_path = None
        if io_mode == 'file':
            try:
                input_path = os.path.join(self.binary._dir, cast(str, input_file)) if input_file is not None else "" # type: ignore
                output_path = os.path.join(self.binary._dir, cast(str, output_file)) if output_file is not None else "" # type: ignore
                if input_path:
                    with open(input_path, 'wb') as f:
                        f.write(case.input_data())
                    # Create case-insensitive symlinks so student code can open
                    # the input file regardless of the case used in fopen().
                    _create_case_symlinks(input_path)
            except OSError as e:
                log.error(f'Failed to set up input file {input_path}: {str(e)}')
                result.result_flag = Result.IE
                result.feedback = f'Cannot create input file: {str(e)}'
                return result

        input_file_io = case.input_data_io() if io_mode == 'std' else None

        try:
            self._launch_process(case, io_mode, input_file , output_file, input_file_io)
        except ValueError as e:
            log.error(f'Failed to launch process: {str(e)}')
            result.result_flag = Result.IE
            result.feedback = str(e)
            if input_path and os.path.exists(input_path):
                try:
                    os.unlink(input_path)
                except OSError as e:
                    log.warning(f'Failed to clean up input file {input_path}: {str(e)}')
            return result

        error = self._interact_with_process(case, result, io_mode, output_path)

        process = self._current_proc
        assert process is not None
        self.populate_result(error, result, process)

        check = self.check_result(case, result)

        if not isinstance(check, CheckerResult):
            check = CheckerResult(check, case.points if check else 0.0)

        result.result_flag |= [Result.WA, Result.AC][check.passed]
        result.points = check.points
        result.feedback = check.feedback or result.feedback
        result.extended_feedback = check.extended_feedback or result.extended_feedback

        if input_path and os.path.exists(input_path):
            try:
                _cleanup_case_symlinks(input_path)
                os.unlink(input_path)
            except OSError as e:
                log.warning(f'Failed to clean up input file {input_path}: {str(e)}')
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError as e:
                log.warning(f'Failed to clean up output file {output_path}: {str(e)}')

        case.free_data()
        return result

    def populate_result(self, error: bytes, result: Result, process: TracedPopen) -> None:
        self.binary.populate_result(error, result, process)

    def check_result(self, case: TestCase, result: Result) -> CheckerOutput:
        # If the submission didn't crash and didn't time out, there's a chance it might be AC
        # We shouldn't run checkers if the submission is already known to be incorrect, because some checkers
        # might be very computationally expensive.
        # See https://github.com/DMOJ/judge-server/issues/170
        checker = case.checker()
        # checker is a `partial` object, NOT a `function` object
        if not result.result_flag or getattr(checker.func, 'run_on_error', False):
            try:
                check = checker(
                    result.proc_output,
                    case.output_data(),
                    submission_source=self.source,
                    judge_input=LazyBytes(case.input_data),
                    point_value=case.points,
                    case_position=case.position,
                    batch=case.batch,
                    submission_language=self.language,
                    binary_data=case.has_binary_data,
                    execution_time=result.execution_time,
                    problem_id=self.problem.id,
                    case=case,
                    result=result,
                )
            except UnicodeDecodeError:
                # Don't rely on problemsetters to do sane things when it comes to Unicode handling, so
                # just proactively swallow all Unicode-related checker errors.
                return CheckerResult(False, 0, feedback='invalid unicode')
        else:
            # Solution is guaranteed to receive 0 points
            check = False

        return check

    def _launch_process(self, case: TestCase, io_mode: str, input_file: str, output_file: str, input_file_io) -> None:
        if io_mode == "file":
            self._current_proc = self.binary.launch(
                time=self.problem.time_limit,
                memory=self.problem.memory_limit,
                symlinks=case.config.symlinks,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                wall_time=case.config.wall_time_factor * self.problem.time_limit,
                input_file=input_file,
                output_file=output_file,
            )
        else:
            self._current_proc = self.binary.launch(
                time=self.problem.time_limit,
                memory=self.problem.memory_limit,
                symlinks=case.config.symlinks,
                stdin=input_file_io or subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                wall_time=case.config.wall_time_factor * self.problem.time_limit,
            )

    def _interact_with_process(self, case: TestCase, result: Result, io_mode: str, output_path: Optional[str]=None) -> bytes:
        process = self._current_proc
        assert process is not None
        try:
            result.proc_output, error = process.communicate(
                None, outlimit=case.config.output_limit_length, errlimit=1048576
            )
        except OutputLimitExceeded:
            log.warning('Output limit exceeded')
            error = b''
            process.kill()
        finally:
            process.wait()

        if io_mode == 'file' and output_path:
            # Try the exact output path first.
            if not os.path.exists(output_path):
                # The expected output file was not found at the exact path.
                # Look for a file with the same name but different case.
                dir_name = os.path.dirname(output_path) or '.'
                base_name = os.path.basename(output_path)
                # Glob for case-insensitive match, excluding symlinks we created.
                pattern = os.path.join(dir_name, _case_glob_pattern(base_name))
                matches = [p for p in glob.glob(pattern) if not os.path.islink(p)]
                if matches:
                    # Found a file with wrong case – rename it so grading can proceed.
                    os.rename(matches[0], output_path)
                else:
                    # No output file at all – the student used a completely wrong filename.
                    log.warning(f'Output file not found (expected {output_path})')
                    result.result_flag = Result.WA
                    result.feedback = f'Wrong filename: expected "{base_name}"'
                    result.proc_output = b''
            if os.path.exists(output_path):
                try:
                    with open(output_path, 'rb') as f:
                        result.proc_output = case._normalize(f.read())
                except OSError as e:
                    log.error(f'Cannot read output file {output_path}: {str(e)}')
                    result.result_flag = Result.IE
                    result.feedback = f'Cannot read output file: {str(e)}'
                    result.proc_output = b''

        return error

    def _generate_binary(self) -> BaseExecutor:
        binary = executors[self.language].Executor(
            self.problem.id,
            self.source,
            hints=self.problem.config.hints or [],
            unbuffered=self.problem.config.unbuffered,
            cached=True,
        )
        return binary


def _cleanup_case_symlinks(path: str) -> None:
    """Remove all symlinks created as case-insensitive helpers for *path*."""
    base = os.path.basename(path)
    dir_name = os.path.dirname(path) or '.'
    for entry in os.listdir(dir_name):
        entry_path = os.path.join(dir_name, entry)
        if os.path.islink(entry_path) and os.path.basename(os.readlink(entry_path)) == base:
            os.unlink(entry_path)


def _create_case_symlinks(path: str) -> None:
    """Create symlinks for every case permutation of the filename *path*.
    This lets student code open() the input file regardless of the casing
    used (lower, UPPER, Mixed, etc.). Symlinks are created as relative
    links inside the same directory and are cleaned up after grading.
    """
    dir_name = os.path.dirname(path) or '.'
    base_name = os.path.basename(path)

    # Separate basename into stem and extension.
    dot = base_name.rfind('.')
    if dot != -1:
        stem = base_name[:dot]
        ext = base_name[dot:]
    else:
        stem = base_name
        ext = ''

    # Collect positions of alphabetic characters in the stem.
    alpha_pos = [(i, ch) for i, ch in enumerate(stem) if ch.isalpha()]

    # If the filename has too many letters the combinatorial explosion
    # isn't worth it – fall back to the three most common patterns.
    MAX_COMBINATIONS = 256  # 2^8
    if len(alpha_pos) > 8:
        _try_symlink(dir_name, base_name, stem.lower() + ext)
        _try_symlink(dir_name, base_name, stem.upper() + ext)
        _try_symlink(dir_name, base_name, stem.capitalize() + ext)
        return

    for bits in itertools.product((0, 1), repeat=len(alpha_pos)):
        chars = list(stem)
        for (idx, _), bit in zip(alpha_pos, bits):
            chars[idx] = chars[idx].upper() if bit else chars[idx].lower()
        variant = ''.join(chars) + ext
        if variant != base_name:
            _try_symlink(dir_name, base_name, variant)


def _try_symlink(dir_name: str, target: str, link_name: str) -> None:
    link_path = os.path.join(dir_name, link_name)
    if not os.path.exists(link_path) and not os.path.islink(link_path):
        os.symlink(target, link_path)


def _case_glob_pattern(filename: str) -> str:
    """Turn a filename into a glob pattern that matches any casing
    by replacing every letter with a character class […].
    """
    result: list[str] = []
    for ch in filename:
        if ch.isalpha():
            result.append(f'[{ch.lower()}{ch.upper()}]')
        else:
            result.append(ch)
    return ''.join(result)