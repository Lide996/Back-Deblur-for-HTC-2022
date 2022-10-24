# The Limited-Angle CT Challenge

#### Wang Jianyu *, Wang Rongqian *, Liu Xintong *, Lin Guochang *, Chen Fukai  *, Cai Lide ** 

#### * Yau Mathematical Science Center, Tsinghua University, Beijing, China

#### ** Department of Mathematical Science, Tsinghua University, Beijing, China 

### Algorithm Introductions

The HTC2022 is about limited-angle computational tomography with data set with limited sinogram from 90-degree to 10-degree. 

The major challenge of this image reconstruction problem are as follows

- Due to the extremely limited probing degree, the obtained real-life data inevitably miss the information of the wavefront set of the singular support of the obstacle. This part of theory is completed by Todd Quinto using microlocal analysis since the late 1980s. It shows the incapability of Back-projection method, even if the measurement is dealt with carefully.
- Due to the ill-posedness induced by the loss of information, measurement error is amplified tremendously in the inversion process, thus suitable regularization technique should be taken. 

Concerning the above challenges, we propose three algorithms:  

- Back-Projection + Image-Deblurring Network, written in matlab, with usage of the Astra toolbox

  The learned deblurring algorithm by trained Back-propagation data produce competitive image.

- FISTA-TV + GAN, written in python, with usage of ...... Toolbox

  The Generative model push forward the remedies of TV-regularization.

- Least Square misfit + Sparse regularization, written in matlab, with usage of Astra toolbox

  The Sparsity-promoting algorithm exploit the structural information of the problem. 

Each one of the algorithms is contained in a separate file.

### Installation instructions, Usage instructions        

- Matlab users: Please specify any toolboxes used.      
- Python users: Please specify any modules used. If you use Anaconda, please add to the repository an environment.yml file capable  of creating an environment than can run your code ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file)). Otherwise, please add a requirements.txt file generated with `pip freeze` ([instructions](https://stackoverflow.com/questions/14684968/how-to-export-virtualenv)) 

### Examples

- Show few examples.