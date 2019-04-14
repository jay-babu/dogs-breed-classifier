from flask import Flask

import dogs_file_organizer
from dogs_file_renamer import breed_labeller

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/dogs_file_rename')
def rename():
    breed_labeller()
    return "Files have been renamed!"


@app.route('/dogs_file_organizer')
def copier():
    dogs_file_organizer.organizer()
    dogs_file_organizer.folder_mover()
    return "Files have been copied successfully and folder has been moved"


if __name__ == '__main__':
    app.run()
