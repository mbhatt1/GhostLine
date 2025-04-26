# GhostLine Playbooks

Author your own call flows in YAML.  
Copy `sample_password_reset.yaml` and tweak:

* Use only stage names in `data.py`.
* Leave `custom_prompt` blank to inherit the default.
* `success_regex` ends the run and flips `outcome = 'compromised'`.

Open a PR to share new playbooks!