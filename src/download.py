from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi

TITANIC_DATA_FOLDER = "data/titanic"

api = KaggleApi()
api.authenticate()


def main():
    # TODO import API_TOKEN into .kaggle/kaggle.json. chmod 600 ~/.kaggle/kaggle.json
    api.competition_download_files("titanic")

    # extract zip
    zf = ZipFile("titanic.zip")
    zf.extractall(TITANIC_DATA_FOLDER)  # save files in selected folder
    zf.close()


if __name__ == "__main__":
    main()
