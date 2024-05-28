import json
from pathlib import Path
import time



class DataFormatter:

    def __init__(self,filename):
        self.filename = filename
    
    def data_formatter(self):

    # Your JSON data
        script_dir = Path(__file__).resolve().parent
        file_path = script_dir / self.file_name
        
        with open(file_path, "r", encoding="utf-8") as output:
            jsonObj = list(output)

        with open("output.json", "w", encoding="utf-8") as output:

            for obj in jsonObj:
                parsed_data = json.loads(obj)
                key_set = {}
                for key, value in parsed_data.items():

                    chord = key
                    print(f"{key}")

                    for chord_data in value:

                        print("Positions:", chord_data['positions'])
                        print("Fingerings:", chord_data['fingerings'])
                        for i,info in enumerate(chord_data['positions']):
                            print(f"Position Index {i+1}:", chord_data['positions'][i])
                        for i,info in enumerate(chord_data['fingerings'][0]):
                            print(f"Finger Index {i+1}:", chord_data['fingerings'][0][i])
                        
                        key_set = {
                            chord : {
                                "positions": {
                                    f"s{i}" : chord_data['positions'][i]
                                },
                                "fingerings": {
                                    f"f{i}" : chord_data['fingerings'][0][i]
                                }
                            }
                            
                        }


                    time.sleep(2)




def main():

    DF = DataFormatter(filename="all-chord.json")

    DF.data_formatter()



if __name__ == '__main__':
    main()