import os
import shutil


def organizer():
    print(os.getcwd())
    # os.chdir("/Users/Jay/Desktop/Dogs/DogsBreedClassifier/")
    original_dataset_dir = 'static'

    base_dir = 'train/data'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')

    validation_dir = os.path.join(base_dir, 'validation')

    test_dir = os.path.join(base_dir, 'test')

    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    # Train folder breeds
    train_bulldog_dir = os.path.join(train_dir, 'bulldog')

    train_german_shepherd_dir = os.path.join(train_dir, 'german_shepherd')

    train_golden_retriever_dir = os.path.join(train_dir, 'golden_retriever')

    train_husky_dir = os.path.join(train_dir, 'husky')

    train_poodle_dir = os.path.join(train_dir, 'poodle')

    os.mkdir(train_bulldog_dir)
    os.mkdir(train_german_shepherd_dir)
    os.mkdir(train_golden_retriever_dir)
    os.mkdir(train_husky_dir)
    os.mkdir(train_poodle_dir)

    # Validation folder breeds
    validation_bulldog_dir = os.path.join(validation_dir, 'bulldog')

    validation_german_shepherd_dir = os.path.join(validation_dir, 'german_shepherd')

    validation_golden_retriever_dir = os.path.join(validation_dir, 'golden_retriever')

    validation_husky_dir = os.path.join(validation_dir, 'husky')

    validation_poodle_dir = os.path.join(validation_dir, 'poodle')

    os.mkdir(validation_bulldog_dir)
    os.mkdir(validation_german_shepherd_dir)
    os.mkdir(validation_golden_retriever_dir)
    os.mkdir(validation_husky_dir)
    os.mkdir(validation_poodle_dir)

    # Test folder breeds
    test_bulldog_dir = os.path.join(test_dir, 'bulldog')

    test_german_shepherd_dir = os.path.join(test_dir, 'german_shepherd')

    test_golden_retriever_dir = os.path.join(test_dir, 'golden_retriever')

    test_husky_dir = os.path.join(test_dir, 'husky')

    test_poodle_dir = os.path.join(test_dir, 'poodle')

    os.mkdir(test_bulldog_dir)
    os.mkdir(test_german_shepherd_dir)
    os.mkdir(test_golden_retriever_dir)
    os.mkdir(test_husky_dir)
    os.mkdir(test_poodle_dir)

    train_percentage = .7
    validation_percentage = .2
    test_percentage = .1

    for breed in os.listdir(original_dataset_dir):
        if breed != '.DS_Store':
            for sub_type in os.listdir(os.path.join(original_dataset_dir, breed)):
                if sub_type != '.DS_Store':
                    subtypes = os.listdir(os.path.join(original_dataset_dir, breed, sub_type))
                    i = 0
                    for name in subtypes:
                        if name != '.DS_Store':
                            train = int(len(subtypes) * train_percentage)
                            validation = int(len(subtypes) * validation_percentage + train)
                            test = int(len(subtypes) * test_percentage + validation)
                            if breed == 'bulldog':
                                if i < train:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(train_bulldog_dir, name))
                                elif i < validation:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(validation_bulldog_dir, name))
                                elif i < test:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(test_bulldog_dir, name))
                            elif breed == 'german_shepherd':
                                if i < train:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(train_german_shepherd_dir, name))
                                elif i < validation:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(validation_german_shepherd_dir, name))
                                elif i < test:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(test_german_shepherd_dir, name))
                            elif breed == 'golden_retriever':
                                if i < train:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(train_golden_retriever_dir, name))
                                elif i < validation:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(validation_golden_retriever_dir, name))
                                elif i < test:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(test_golden_retriever_dir, name))
                            elif breed == 'husky':
                                if i < train:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(train_husky_dir, name))
                                elif i < validation:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(validation_husky_dir, name))
                                elif i < test:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(test_husky_dir, name))
                            elif breed == 'poodle':
                                if i < train:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(train_poodle_dir, name))
                                elif i < validation:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(validation_poodle_dir, name))
                                elif i < test:
                                    shutil.copyfile(os.path.join(original_dataset_dir, breed, sub_type, name),
                                                    os.path.join(test_poodle_dir, name))
                            i += 1


def folder_mover():
    base_dir = 'train/data'
    original_dataset_dir = 'static'
    shutil.move(base_dir, original_dataset_dir)


if __name__ == '__main__':
    organizer()
    folder_mover()
