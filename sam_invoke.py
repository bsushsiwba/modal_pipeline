import os
import time
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

from samutils import samutils_segment

base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "shirt": "shirt",
            "trouser": "trouser",
        }
    )
)
print("SAM ready")

# wait for process_sam.txt to be created
while not os.path.exists("process_sam.txt"):
    time.sleep(0.1)
# delete process_sam.txt
os.remove("process_sam.txt")

samutils_segment(base_model)

# create sam_complete.txt to signal completion
with open("sam_complete.txt", "w") as f:
    f.write("done")
