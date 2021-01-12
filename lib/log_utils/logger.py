from pathlib import Path
import sys
if sys.version_info.major == 2: # Python 2.x
    from StringIO import StringIO as BIO
else:                           # Python 3.x
    from io import BytesIO as BIO
from torch.utils.tensorboard import SummaryWriter


class PrintLogger(object):

    def __init__(self):
        """Create a summary writer logging to log_dir."""
        self.name = 'PrintLogger'

    def log(self, string):
        print (string)

    def close(self):
        print ('-'*30 + ' close printer ' + '-'*30)


class Logger(object):

    def __init__(self, log_dir, seed, create_model_dir=True):
        """Create a summary writer logging to log_dir."""
        self.seed      = int(seed)
        self.log_dir   = Path(log_dir)
        self.model_dir = Path(log_dir) / 'model'
        self.log_dir.mkdir  (parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_dir = self.log_dir
        self.logger_path = self.log_dir / 'seed-{:}.log'.format(self.seed)
        self.logger_file = open(self.logger_path, 'w')

        self.tensorboard_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.tensorboard_dir))

    def __repr__(self):
        return ('{name}(dir={log_dir}, writer={writer})'.format(name=self.__class__.__name__, **self.__dict__))

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string); sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write('{:}\n'.format(string))
            self.logger_file.flush()
