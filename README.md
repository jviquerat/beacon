# beacon

<p align="center">
  <img align="right" width="350" alt="logo" src="msc/logo.png">
</p>

This repository proposes benchmark cases for DRL-based flow control. The considered cases are voluntarily limited in CPU requirements, so they can be used for prototyping on local machines while still presenting realistic flow control aspects.

## `shkadov`

The agent actuates several jets to damp the instabilities of a falling fluid film (original approach was from Belus et al. in <a href="https://aip.scitation.org/doi/10.1063/1.5132378">this paper</a>.) States are the height and mass flow rate upstream of each jet. One episode represents 400 actions, and the training is made on 500000 transitions for 5 jets (approx. 30 mins on a laptop with PPO algorithm).

<p align="center">
  <img width="700" alt="" src="shkadov/save/shkadov.gif">
</p>
