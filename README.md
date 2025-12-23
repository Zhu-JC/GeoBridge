# GeoBridge
 Generating and navigating single cell dynamics viaby a geodesic bridge between nonlinear transcriptional and linear latent manifolds
![Schematic of GeoBridge](./readme/schematic.png)
## Geodesics path of EMT(green)-MET(orange) progression
| original nonlinear manifold | latent linear manifold (EMT) | latent linear manifold (MET) |
|:----------------------------:|:-----------------------------:|:-----------------------------:|
| ![original](./readme/MET_EMT_org.gif) | ![EMT](./readme/EMT_eu.gif) | ![MET](./readme/MET_eu.gif) |
## Geodesics path of Beta(green)-Ec(orange) progression
| original nonlinear manifold | latent linear manifold (Beta) | latent linear manifold (Ec) |
|:----------------------------:|:-----------------------------:|:-----------------------------:|
| ![original](./readme/Beta_Ec_org.gif) | ![EMT](./readme/sc_beta_eu.gif) | ![MET](./readme/sc_ec_eu.gif) |
## Installation
### 1️⃣ Create a new Conda environment
```
conda create -n GeoBridge python=3.9
conda activate GeoBridge
```
### 2️⃣ Install the FrEIA dependency
pip install git+https://github.com/vislearn/FrEIA.git

### 3️⃣ Clone the GeoBridge repository
git clone https://github.com/Zhu-JC/GeoBridge.git
cd GeoBridge

### 4️⃣ Install additional dependencies
pip install -r requirements.txt
