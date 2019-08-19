Installing
==========

Getting Python
--------------
tetra3 is written for Python 3.7 (and therefore runs on almost any platform) and should work with
most modern Python 3 installations. There are many ways to get Python on your system. The preferred
method is by `downloading Miniconda for Python 3+ 
<https://docs.conda.io/en/latest/miniconda.html>`_. Afterwards, open `Anaconda Prompt` and type::

    conda update conda
    
to ensure you are up to date.

Getting tetra3
--------------
tetra3 is not available on PIP/PyPI (the Python Package Index). Instead you need to get the software
repository from GitHub and place it in the directory where you wish to use it. (I.e. as a folder
next to the python project you are writing.)

The quick way
^^^^^^^^^^^^^
Go to `the GitHub repository <https://github.com/esa/tetra3>`_, click `Clone or Download` and
`Download ZIP` and extract the tetra3 directory to where you want to use it.

The proper way
^^^^^^^^^^^^^^
To be able to easily download and contribute updates to tetra3 you should install Git. Follow the
instructions for your platform `over here <https://git-scm.com/downloads>`_.

Now open a terminal/CMD window in the directory where you wish to use tetra3 and clone the
GitHib repository::

    git clone "https://github.com/esa/tetra3.git"
    
You should see the tetra3 directory created for you with all neccessary files. Check the status of
your repository by typing::

    cd tetra3
    git status
    
which should tell you that you are on the branch "master" and are up to date with the origin (which
is the GitHub version of tetra3). If a new update has come to GitHub you can update yourself by
typing::

    git pull

If you wish to contribute (please do!) and are not familiar with Git and GitHub, start by creating
a user on GitHub and setting you username and email::

    git config --global user.name "your_username_here"
    git config --global user.email "email@domain.com"

You will now also be able to push proposed changes to the software. There are many good resources
for learning about Git, `the official documentation <https://git-scm.com/doc>`_ which includes the 
reference, a book on Git, and basic videos is a good place to start.

Installing tetra3
-----------------
To install the requirements open a terminal/CMD window in the tetra3 directory and run::

    python setup.py install
    
to install all requirements. Test that everything works by running an example::

    cd examples
    python test_tetra3.py
    
which should print out the solutions for the included test images.
    
If problems arise
-----------------
Please get in touch by `filing an issue <https://github.com/esa/tetra3/issues>`_.