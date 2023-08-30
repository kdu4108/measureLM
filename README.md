# Deployment

## Set-up

### Virtual Environment

```
cd /Users/username/Code/entityLM 
virtualenv entityLM_venv
source entityLM_venv/bin/activate
```
to register Kernel with Jupiter Notebook
```
pip install ipykernel
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
python3 -m ipykernel install --user --name=entityLM_venv
```
 
Check the kernelspec list whether this projects uses the right executable
```
jupyter kernelspec list
```
the kernelspec entries point to kernel.json
```
'/Users/username/code/entityLM/entityLM_venv/bin/python3'
```
the kernel.json should look as follows, i.e. it installs dependencies within the virtualenv
```
{
 "argv": [
  "/Users/username/code/entityLM/entityLM_venv/bin/python3",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "entityLM_venv",
 "language": "python"
}
```
the kernel entry in kernelspecs may be deleted with
```
jupyter kernelspec uninstall entityLM_venv
```
### Activate and deactivate the environment

- to activate:

```
cd /Users/username/Code/entityLM
pip install -e ../entityLM
```

- to deactivate:

```
deactivate
```


## Install sub-dependencies

To develop, you have to execute package instalment in every module directory. This allows importing the directory in other scripts and changes become effective immediately (no further deployment). 
```
cd /Users/username/code/entityLM
pip install -e . entityLM
```

# Github

Setting up new GitHub repository

```
(add .gitignore)
git init
git add .
git commit -m "first commit"
git remote add origin git@github.com:...
git branch -M master 
git push -u origin master
```