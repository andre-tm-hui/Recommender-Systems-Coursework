To use this CARS, run the command:

		python main.py (-t -s -l -e -lr -f -ld -os)

	-t *int* - enables(1)/disables(0) training
	-s *int* - enables(1)/disables(0) saving the trained features/datasets
	-l *int* - enables(1)/disables(0) loading of saved features/datasets
	-e *int* - sets the number of training cycles over the dataset
	-lr *float* - sets the learning rate of the CARS
	-f *int* - sets the number of features used to represent each user/item/context condition
	-ld *float* - sets the regularization term, lambda
	-os *int* - indicates the OS of the system to the program (0 for Windows, 1 for Linux)

Enabling saving will create a directory called "dat", containing all the trained features and datasets.
It is recommended not to use any of the above arguments, apart from "-os". Hence, the recommended commands are:

	Windows:	python main.py -os 0
	Linux:		python3 main.py -os 1

If the "os" argument is not set properly, the console-based UI may become less clean.

Once the training is complete and the results are produced, you will be prompted to login as an existing user.
Enter a number in the range 0-2370 to login.
Once logged in, you have the option of 4 inputs:
	- r *N* - generates a list of the top N predictions, N being an integer in range 1-2269
	- t *T* - sets the number of travellers. Any integer less than 1 will result in initial recommendations
		  being produced
	- p	- returns all of the users' previous ratings, as well as their ratings as predicted by the CARS
	- esc	- ends the program
