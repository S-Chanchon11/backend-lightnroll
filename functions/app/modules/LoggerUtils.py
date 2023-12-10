import csv
import logging
from datetime import datetime, date


class LogUtils:

    def csv_logger(self,data=None):

        header = ['logs','time']

        with open("__log_history__.csv", "a", encoding='utf-8') as output:

            writer = csv.writer(output)

            #writer.writerow(header)

            now = datetime.now()

            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

            log_format = [data,dt_string]

            writer.writerow(log_format)

def main():
    lu = LogUtils()
    lu.csv_logger()

if __name__ == '__main__':
    main()