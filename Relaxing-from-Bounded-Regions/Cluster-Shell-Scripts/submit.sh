# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory="$EPHEMERAL/L96-EBM-Ecape-Rates/Relaxations-from-Bounded-Regions/$NOW"
mkdir -p $run_directory
cp -r $HOME/Instantons/L96-EBM-Escape-Rates/Relaxing-from-Bounded-Regions/ $run_directory
cd $run_directory/Relaxing-from-Bounded-Regions
cp $run_directory/Relaxing-from-Bounded-Regions/Cluster-Shell-Scripts/relax-from-bounded-regions.sh .
qsub relax-from-bounded-regions.sh
