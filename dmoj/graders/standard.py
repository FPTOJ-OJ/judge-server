import logging
import os
import subprocess
from typing import Optional
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
                input_path = os.path.join(self.binary._dir, input_file) if input_file else None
                output_path = os.path.join(self.binary._dir, output_file) if output_file else None
                if input_path:
                    log.debug(f'Creating input file: {input_path}')
                    with open(input_path, 'wb') as f:
                        f.write(case.input_data())
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
                os.unlink(input_path)
            except OSError as e:
                log.warning(f'Failed to clean up input file {input_path}: {str(e)}')
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
                log.debug(f'Cleaned up output file: {output_path}')
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
            try:
                with open(output_path, 'rb') as f:
                    result.proc_output = case._normalize(f.read())
            except OSError as e:
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
        )
        return binary