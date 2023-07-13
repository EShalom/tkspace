.. _developer-guide:

###############
Developer guide
###############

*******************************
How to contribute documentation
*******************************


*******************************
How to contribute functionality
*******************************

************************
How to contribute issues
************************

If you have a constructive suggestion for how ``tkspace`` can be improved, but you are not able to address it yourself for some reason, it is still extremely helpful if you write this up as an issue so it can be picked up by others at a later stage. To write up an issue, go to the ``osipi`` repository on github, select `issues` and write a new one. Make sure to provide sufficient detail so that others can understand and address the issue.
 
Package structure
^^^^^^^^^^^^^^^^^

The ``tkspace`` package currently only includes methods for the forward simulation of 1D and 2D 1 compartment and 2 compartment systems, within the inverse problem method applied for 1D problems of both compartment types.

::

    tkspace.models
    ├── models
    │   ├── onedim
    │   │   ├── cases
    │   │   ├── flow
    │   │   ├── velocity
    │   │   ├── loadcase
    │   │   ├── sampling
    │   │   ├── systeminfo
    │   │   └── fm_utilities
    │   ├── twodim
    │   │   ├── flow
    │   │   └── fm_utilities
    ├── inversion
    │   ├── gradient_descent
    │   ├── cost_functions
    │   └── guess
    ├── plotting 
    │   └── param_comp           
    └── utilities  


