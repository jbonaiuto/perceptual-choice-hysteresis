perceptual-choice-hysteresis
============================

Simulation and analysis code for
> JJ Bonaiuto, A de Berker, S Bestmann<br>
> **Neural hysteresis in competitive attractor models predicts changes in choice bias with non-invasive brain stimulation**<br>
> eLife 2016, https://elifesciences.org/articles/20047

Required python libraries and versions are listed in requirements.txt

Use pip (https://pypi.python.org/pypi/pip) to install the requirements:

    pip install -r requirements.txt

inside a virtual environment (https://pypi.python.org/pypi/virtualenv) to install.


From the src/python directory, run:

    python perceptchoice.experiment.analysis

to analyze human participant data, or:

    python perceptchoice.model.run

to run model simulations, or:

    python perceptchoice.model.analysis

to analyze model simulations.

Data are archived at http://dx.doi.org/10.5061/dryad.r1072

Email jbonaiuto@gmail.com with any questions.
