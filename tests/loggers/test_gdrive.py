import pytest
from time import sleep
from argparse import Namespace
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pytorch_lightning.loggers import GDriveLogger

EXPERIMENT_NAME = 'TestGDriveLogger'
PYDRIVE_SETTINGS_FILE = 'creds/pydrive_settings.yaml'


@pytest.fixture(scope="session")
def drive():
    gauth = GoogleAuth(settings_file=PYDRIVE_SETTINGS_FILE)
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


@pytest.fixture
def remove_experiment_dir_from_gdrive(drive):
    file_list = drive.ListFile(
        {'q': f"'root' in parents and trashed=false and title='{EXPERIMENT_NAME}'"}).GetList()
    for _file in file_list:
        if _file['mimeType'] == 'application/vnd.google-apps.folder':
            _file.Trash()
            break


def test_gspread_logger(remove_experiment_dir_from_gdrive):
    hparams = Namespace(**{
        'drop_prob': 0.2,
        'batch_size': 32,
        'in_features': 28 * 28,
        'learning_rate': 0.001 * 8,
        'optimizer_name': 'adam',
        'data_root': '/data',
        'out_features': 10,
        'hidden_dim': 1000,
    })
    logger = GDriveLogger(name=EXPERIMENT_NAME, version=0,
                          settings_file=PYDRIVE_SETTINGS_FILE,
                          gspread_credentials='creds/gsheets_credentials.json')
    logger.log_hyperparams(hparams)
    for _idx in range(1, 21):
        logger.log_metrics({'loss': 20.0 / _idx, 'acc': _idx / 20.0}, step=_idx)
        if _idx % 3 == 0:
            logger.save()
            logger.log_metrics({'val_loss': 20.0 / _idx, 'mIoU': _idx / 20.0}, step=_idx)
        if _idx % 5 == 0:
            logger.save()
            logger.log_metrics({'test_mIoU': _idx / 20.0, 'test_AP': _idx / 40.0}, step=_idx)
        sleep(1)
    # TODO: find out why number in asserted cells return as strings
    for _idx, _cell in enumerate(logger._ws.range("A4:A23")):
        assert int(_cell.value) == _idx + 1
    for _idx, _cell in enumerate(logger._ws.range("E4:E9")):
        assert int(_cell.value) == (_idx + 1) * 3
    for _idx, _cell in enumerate(logger._ws.range("I4:I6")):
        assert int(_cell.value) == (_idx + 1) * 5