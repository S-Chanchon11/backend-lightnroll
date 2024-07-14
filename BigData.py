import csv

csv_file_path = 'diamonds.csv'  # Path to your CSV file
text_file_path = 'output.txt'   # Path to the output text file

columns = {}

# Read the CSV file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    # Initialize lists for each column
    for field in reader.fieldnames:
        columns[field] = []
    # Populate lists with data
    for row in reader:
        for field in reader.fieldnames:
            if(field=="x" or field=="y" or field=="z"):
                columns[field].append(f"{field}_{row[field]}")
            # elif(field=="cut"):
            #     match(row[field]):
            #         case "Fair": row[field] == 0
            #         case "Good": row[field] == 1
            #         case "Very Good": row[field] == 2
            #         case "Premium": row[field] == 3
            #         case "Ideal": row[field] == 4
            #     columns[field].append(f"{field}_{row[field]}")
            else: #Fair, Good, Very Good, Premium, Ideal
                columns[field].append(row[field])

# Write the output to a text file
with open(text_file_path, 'w') as txtfile:
    for field in reader.fieldnames:
        for value in columns[field]:
            txtfile.write(value + '\n')
