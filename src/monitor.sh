RUNNING=$(squeue | grep "gpu" | wc -l)
PENDING=$(squeue | grep "PD" | wc -l)

USER="pallaoro"

RUNNINGUSR=$(squeue -u $USER | grep "gpu" | wc -l)
PENDINGUSR=$(squeue -u $USER | grep "PD" | wc -l)

echo "Running Processes: $RUNNING ($RUNNINGUSR)"
echo "Pending Processes: $PENDING ($PENDINGUSR)"