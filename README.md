# Reserving_techniques
> This project is created to **Estimate the Ultimate Claims of a company in a particular year**
## install these requiremnets in your environment using the following command:
`pip install -r requirements.txt`

## The following methods are used in this project
1. Chain-Ladder
2. Exp Claims 
3. Bornheutter-Fergueson
4. Cape-cod
5. GLM-3 varaitions
6. Bootstrap- 2 variations
7. Neural networks(coming soon)

###
- *run the run_single.py file* using `python run_single.py` You can change the line of business and company code as you like(more flexibility coming soon).
   - You will create influx of  images :smile:
- *run the run_multiple.py file* using `python run_multiple.py` This will generate results in the form of a dataframe which is pickled, 
   - *run output.py* using `python output.py` to unpickle and extract the results from  `output.xslv`
- *info.pickle* is the binary file containing the results of multiple_companies. It is stored for reference

>Front-end coming soon 

