Revised Graph transformer (Dwivedi, V. P. and Bresson, X. (2020)) for Extreme Event Prediction

Reproducibility:

1 - Adjust the preferred parameters for the dataset parameters and model settings in configs/SU_params.json
2 - Run data_prep.ipynb
3 - Run main.py


	Parameter description:
	
	"update_tiles" recomputes tiles if set to True, otherwise loads previously saved tiles;
	"tiles_per_row" sets the resolution of the graph partitioning, the total number of tiles is the square of this number (rows * columns); (*)
	"extremes only" selects slope units associated with non-zero targets only if set to True, otherwise uses the full dataset; (*)
	"full_graphs" graphs are computed as complete graphs (fully connected) if set to True, otherwise edges are present for adjacent slope units only; (*)
	  

(*) update tiles set to True is required to apply the changes

Note: The dataset analysed in the present study is currently not publicly available in agreement with the third party data provider.