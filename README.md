# Inpaint training on a new dataset

Make sure you have a GPU based machine to speed up the training process. I have seleced celebahq dataset as the example.
Instructions are based on a GNU/Linux OS.


1.Clone the repository

`git clone https://github.com/aiarcade/inpainttraining.git`

2.Install the required python library to the virtual env.

`cd inpainttraining`
`pip install -r requirements.txt' `

3.Download the celebahq dataset from the   [google drive](https://drive.google.com/file/d/1O89DVCoWsMhrIF3G8-wMOJ0h7LukmMdP/view?usp=sharing) and place it inside the
directory inpainttraining.

4.Preprocess the downloaded dataset using the command.

`sh celebahq_data_prepare.sh`

5.Create the random mask  the dataset using the command.

`sh celebahq_gen_masks.sh`

6.Run the model training process using the command.

`python3 train.py -cn lama-fourier-celeba data.batch_size=10`

Once the training is completed , the model file will be avaialable in experiments directory. The same procedure can be followed for any dataset.
