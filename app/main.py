from flask import Flask

import dogs_file_renamer

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/dogs_file_rename')
def rename():
    dogs_file_renamer.breed_labeller()
    return "Files have been renamed!"


if __name__ == '__main__':
    app.run()
