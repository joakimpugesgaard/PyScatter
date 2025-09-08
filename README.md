# PyScatter

A Python library for calculating and visualizing interactions between light and spherical particles, specifically plane waves and Laguerre-Gaussian beams focused with an aplanatic lens. 
It can visualize complicated beam profiles and their multipolar content and calculate light-matter interactions using Generalized Lorentz-Mie Theory. With this formalism, the resulting electric fields from scattering and absorption processes can be displayed along with cross-sections.
Finally, it extends the T-matrix based software "treams" to work with the complex focused beams defined with PyScatter. This allows a user to apply the illuminatio from PyScatter with the more complex scatterer configurations available with treams.

The full documentation can be read here: https://joakimpugesgaard.github.io/PyScatter/index.html


## Installation
```bash
git clone https://github.com/joakimpugesgaard/PyScatter.git
cd PyScatter
pip install -r requirements.txt
```

## Getting started

In the 'examples' folder, Jypyter Notebooks for each module can be accessed. These explain and demonstrate the key methods of the class. This is a good starting point to learn how to call the various functions and extract relevant data or plots.
For more in-depth problems, the source code in 'src' can be accessed.

