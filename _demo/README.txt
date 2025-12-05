// when you first enter GUI, the DIAGNOSE button should be disabled because no model has been uploaded. 

I. Download the *demo folder. 

II. Backend Activation
1. Navigate to the downloaded folder in terminal, some useful commands include
	a. cd directory_name(change to directory)
	b. cd .. (go back to the last directory)
	c. ls (list all the subelements in the current directory) 
2. Once you are in the right folder, type "python app.py"(you don't need the quotations). This will activate the Python backend for you; 
3. If at some point, you're trying to terminate the backend, hold ctrl+c in the terminal. 

III. Guidance to use this GUI
1. Locate the .html file in your finder, double click, then you should see the GUI pops up in your default web browser; 
2. Click "Admin Model Upload"; 
3. Enter the code "cerviGOAT"(case-sensitive) -- this is for security reasons;
4. Now you are at an interface that asks you to upload your model, upload the "trained_cervical_classifier.pkl" model from this folder;
5. Now you should be able to upload your images, and see the corresponding results, have fun! 

IV. Results Interpretation: 
1. Result: precancerous, cancerous and normal; 
2. Confidence level: The confidence level refers to the probability(model outputs probability distribution for all classes) assigned to the class with the highest likelihood.