RayPy2D Simple 2D Optics Simulation
===============================
author: Tobias Schoch

Overview
--------

A simple simulation for 2D arrangments of basic optical elements


Change-Log
----------
##### 0.2.0
* fixed parabolic mirror flipped
* flipped argument for aperture outline
* aperture outline drawing
* fixed optical path and diffraction grating
* fixed mirror and parabolic mirror
* fixed plotting of parabolic mirror
* bring back basic functionality
* work on plotting tests
* lens and aperture can be plotted again
* more on plotting modularization
* added a Rays object for facilitating the rays interpretation, some work on modularization of plotting

##### 0.1.2
* Renamed ImagePath -> OpticalPath
* fixed bug in ray propagation with zero angle in parabolic mirror

##### 0.1.1
* fixed parabolic surface of parabolic mirror
* rename module to raypy2d
* rename to raypy2d as raypy is already on pypi
* changed package name to raypy

##### 0.1.0
* imageing path with diffraction grating working but ParabolicMirror still flat
* diffraction grating is working
* object with ray fans
* draw arcs for lenses
* imaging path basic functionality
* update readme
* initial commit, basic raytrace functional

##### 0.0.1
* initial version


Installation / Usage
--------------------

To install use pip:

    pip install git@github.com:toschoch/python-raypy.git


Or clone the repo:

    git clone git@github.com:toschoch/python-raypy.git
    python setup.py install
    
Contributing
------------

TBD

Example
-------

TBD