import subprocess 
import os

reps = 10

folder = "../smtlib-fsm/linear-p1/"

outputfile = open("cvc5-lin-p1.txt", "w")

for file in os.listdir(folder):
    # print(file)
    command = "cvc5 "+folder+file+" --stats"
    print(command)
    for i in range(reps):
        outputfile.write("\nrep "+ str(i)+" file "+folder+file)
        process = subprocess.run([command], shell=True, capture_output=True, text=True)
        # Store the output
        output = process.stdout
        error_output = process.stderr
        # Print the outputs
        # print("Standard Output:")
        # print(output)
        # print("Error Output:")
        # print(error_output)
        outputfile.write(output)
        outputfile.write(error_output)


        
outputfile.close()
