# AI_Unsupervised_Learning
AI Final Project on Unsupervised Learning (SOM)


Self Organizing Map
==============================================================

This application uses a self organizing map to visualize 
economic activity data from the World Bank.

Instructions
---------------------------------------------------------------

You will need python3, pandas, numpy and tkinter.
After installing the libraries, simply run the somCountries.py file.
You can edit SOM configuration in the somCountries.py file.
The dataset can be found on the countries_data file.

Files
---------------------------------------------------------------

somCountries.py => main file used to create a self organizing map
and display it using a hexagonal grid.

hexgrid.py => contains classes used to create a hexagonal grid
using tkinter. it is easy to change and the SomGrid class is 
particular to this specific aplication, while the other classes
are used to draw a hexagonal grid.

som.py => This is the self organizing map. The file is properly 
commented for better understanding.

countries_data => csv file containing economic data from 124 countries.
This data was collected from the World Bank.

