import os

base_dir = "/Users/Jay/Desktop/Dogs/DogsBreedClassifier/static"
bulldog = os.path.join(base_dir, "bulldog")
german_shepherd = os.path.join(base_dir, "german_shepherd")
golden_retriever = os.path.join(base_dir, "golden_retriever")
husky = os.path.join(base_dir, "husky")
poodle = os.path.join(base_dir, "poodle")


def breed_labeller():
    counter = 0
    for dog_type in os.listdir(bulldog):
        if dog_type != ".DS_Store":
            current_dog_type = os.path.join(bulldog, dog_type)
            os.chdir(current_dog_type)
            for file in os.listdir(current_dog_type):
                file_path = os.path.join(current_dog_type, file)
                dog_type_formatted = dog_type.split()[1]
                ext = os.path.splitext(file_path)[1]
                os.rename(file, os.path.join(f"{counter}.{dog_type_formatted}{ext}"))
                counter += 1
        counter = 0

    for dog_type in os.listdir(german_shepherd):
        if dog_type != ".DS_Store":
            current_dog_type = os.path.join(german_shepherd, dog_type)
            os.chdir(current_dog_type)
            for file in os.listdir(current_dog_type):
                file_path = os.path.join(current_dog_type, file)
                dog_type_formatted = dog_type.split()[1]
                ext = os.path.splitext(file_path)[1]
                os.rename(file, os.path.join(f"{counter}.{dog_type_formatted}{ext}"))
                counter += 1
        counter = 0

    for dog_type in os.listdir(golden_retriever):
        if dog_type != ".DS_Store":
            current_dog_type = os.path.join(golden_retriever, dog_type)
            os.chdir(current_dog_type)
            for file in os.listdir(current_dog_type):
                file_path = os.path.join(current_dog_type, file)
                dog_type_formatted = dog_type.split()[1]
                ext = os.path.splitext(file_path)[1]
                os.rename(file, os.path.join(f"{counter}.{dog_type_formatted}{ext}"))
                counter += 1
        counter = 0

    for dog_type in os.listdir(husky):
        if dog_type != ".DS_Store":
            current_dog_type = os.path.join(husky, dog_type)
            os.chdir(current_dog_type)
            for file in os.listdir(current_dog_type):
                file_path = os.path.join(current_dog_type, file)
                dog_type_formatted = dog_type.split()[1]
                ext = os.path.splitext(file_path)[1]
                os.rename(file, os.path.join(f"{counter}.{dog_type_formatted}{ext}"))
                counter += 1
        counter = 0

    for dog_type in os.listdir(poodle):
        if dog_type != ".DS_Store":
            current_dog_type = os.path.join(poodle, dog_type)
            os.chdir(current_dog_type)
            for file in os.listdir(current_dog_type):
                file_path = os.path.join(current_dog_type, file)
                dog_type_formatted = dog_type.split()[1]
                ext = os.path.splitext(file_path)[1]
                os.rename(file, os.path.join(f"{counter}.{dog_type_formatted}{ext}"))
                counter += 1
            counter = 0


if __name__ == '__main__':
    breed_labeller()
