# beacon

<p align="center">
  <img align="right" width="350" alt="logo" src="beacon/msc/logo.png">
</p>

This repository proposes benchmark cases for DRL-based flow control. The considered cases are voluntarily limited in CPU requirements, so they can be used for prototyping on local machines while still presenting realistic flow control aspects. If you use this library in the context of your research work, please consider citing:

TO COMPLETE

## `shkadov`

The agent actuates several jets to damp the instabilities of a falling fluid film (original approach was from Belus et al. in <a href="https://aip.scitation.org/doi/10.1063/1.5132378">this paper</a>.) States are the mass flow rates of the fluid upstream of each jet. One episode represents 400 actions, and the training is made on 200000 transitions for 1 to 10 jets.

<p align="center">
  <img width="700" alt="" src="beacon/msc/shkadov.gif">
</p>

## `sloshing`

The agent controls the acceleration of a tank containing a fluid in order to damp the sloshing movement initiated during an excitation pahse. State vector is a downsampled mass flow rate vector. One episode represents 200 actions, and the training is made on 200000 transitions.

<p align="center">
  <img width="700" alt="" src="beacon/msc/sloshing.gif">
</p>
