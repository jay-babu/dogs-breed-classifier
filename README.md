# Dogs Breed Classifier

### Dogs_file_renamer
This python file was used to rename all different breeds of dogs to format this format:\
Number (This was an arbitrary integer). Subtype of breed. File Extension

The reason each breed has subtypes is so the model is able to train on various types of breeds 
and not see them some for the first time. For example, the model should train on white bulldogs, so it is able to expect
them in the model, instead of there being a risk of it being seen the first time in the test.

### Dogs_file_organizer
The purpose of this file was to get 70% of each subtype of each breed and place it in the training folder, 20% in the
validation folder, 10% in the test folder.