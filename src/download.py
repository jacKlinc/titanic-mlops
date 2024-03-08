from zipfile import ZipFile
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

TITANIC_DATA_FOLDER = "data/titanic"

api = KaggleApi()
api.authenticate()


def delete_file(filename: str):
    try:
        pathlib.Path(filename).unlink()
        print("file deleted")
    except:
        print("File doesn't exist")


def main():
    # TODO import API_TOKEN into .kaggle/kaggle.json. chmod 600 ~/.kaggle/kaggle.json
    api.competition_download_files("titanic")
    print("file downloaded")

    # extract zip
    TITANIC_FILE = "titanic.zip"
    zf = ZipFile(TITANIC_FILE)
    zf.extractall(TITANIC_DATA_FOLDER)  # save files in selected folder
    zf.close()
    print("files extracted")
    
    # delete zip
    delete_file(TITANIC_FILE)


if __name__ == "__main__":
    main()
