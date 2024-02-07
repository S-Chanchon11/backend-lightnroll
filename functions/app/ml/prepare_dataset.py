import librosa
import os
import json
import numpy as np
from PCP import PitchClassProfiler
from HPCP import hpcp, calculate_hpcp, my_enhanced_hpcp, myhpcp

DATASET_PATH = "/Users/snow/backend-lightnroll/functions/app/ml/guitar_chord/Training"
DATA_TEST_PATH = "/Users/snow/backend-lightnroll/functions/app/ml/guitar_chord/Test1"
# JSON_PATH = "data_maj_chord_v1.json"
JSON_PATH = "data_enhance_ver.json"
JSON_PATH_TEST = "test.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec. of audio

# Extend the JSONEncoder class
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def preprocess_data_cnn(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)

                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))
            
    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def preprocess_data_pcp(dataset_path, json_path):
    """Extracts pitch class from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save pitchs
        :return:
        """

    # dictionary to store mapping, labels, and pitch classes
    data = {
        "mapping": [],
        "labels": [],
        "pitch": [],
        "order":[]
    }

    # loop through all chord sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        #          C       C#      D      D#     E      E#     F     G      G#     A      A#    B
        fref_0 = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87]
        fref_1 = [32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74]
        # fref = fref_0 + fref_1
        
        # ensure we're processing a sub-folder level
        if dirpath is not dataset_path:
            
            # save chord label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            
            # process all audio files in chord sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                file_name= file_path.split("/")[-1]
                file_name2=file_name.split(".")[0]
                data["order"].append(file_name2)
                # process all segments of audio file
                # data["pitch"].append(myhpcp(audio_path=file_path, fref=fref_0[i-1]))
                data["pitch"].append(my_enhanced_hpcp(audio_path=file_path, fref=fref_0[i-1], pcp_num=12))
                data["labels"].append(i - 1)
                # print(int(float(data["order"][0])))
                print("{}, segment:{}".format(file_path, 1))
        else:
            print("oups")
    

    # save pitch classes to json file
    with open(json_path, "w") as fp:
        #print(type(data))
        json.dump(data,fp, indent=4, cls=NumpyEncoder)
        

def length_check(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = data["pitch"]
    arr = [
                0.06547342985868454,
                0.07053419202566147,
                0.0491626039147377,
                0.03711896762251854,
                0.06277700513601303,
                0.06436990201473236,
                0.04378801956772804,
                0.02222456969320774,
                0.014001060277223587,
                0.024316927418112755,
                0.026309847831726074,
                0.04102746769785881,
                0.04991992935538292,
                0.045685648918151855,
                0.05821360647678375,
                0.04726145416498184,
                0.0369495153427124,
                0.024302955716848373,
                0.2255914956331253,
                0.11306135356426239,
                0.046589817851781845,
                0.011776238679885864,
                0.004022623412311077,
                0.007214454934000969,
                0.02835649624466896,
                0.024032147601246834,
                0.013534803874790668,
                0.02621951326727867,
                0.03989247977733612,
                0.04242663085460663,
                0.034963659942150116,
                0.015568719245493412,
                0.01877296343445778,
                0.016039306297898293,
                0.018217768520116806,
                0.0166922714561224,
                0.03528561443090439,
                0.09257622808218002,
                0.07755651324987411,
                0.06295672804117203,
                0.040004413574934006,
                0.0304169449955225,
                0.038795944303274155,
                0.025694778189063072,
                0.025128304958343506,
                0.03870311751961708,
                0.04128745198249817,
                0.047598328441381454,
                0.0490710623562336,
                0.04232988879084587,
                0.037733227014541626,
                0.06948300451040268,
                0.12950143218040466,
                0.08411747217178345,
                0.07327226549386978,
                0.058842454105615616,
                0.05613012611865997,
                0.03810843452811241,
                0.02177707478404045,
                0.02654387429356575,
                0.025045912712812424,
                0.03957463055849075,
                0.04188794270157814,
                0.031022246927022934,
                0.02335742488503456,
                0.028778694570064545,
                0.03327224776148796,
                0.04152422025799751,
                0.04062814265489578,
                0.02565860189497471,
                0.021991878747940063,
                0.021317414939403534,
                0.02583743818104267,
                0.030493084341287613,
                0.023201584815979004,
                0.022810032591223717,
                0.030062619596719742,
                0.02495158277451992,
                0.024874163791537285,
                0.02029280737042427,
                0.027951030060648918,
                0.02928757853806019,
                0.03099340945482254,
                0.03739848732948303,
                0.028092198073863983,
                0.021749740466475487,
                0.017824187874794006,
                0.027016127482056618,
                0.03275052830576897,
                0.018044255673885345,
                0.020250361412763596,
                0.02906169928610325,
                0.03463632985949516,
                0.03507326915860176,
                0.028187323361635208,
                0.02147616259753704,
                0.014358201064169407,
                0.01307184249162674,
                0.015475309453904629,
                0.019634658470749855,
                0.015570209361612797,
                0.016652867197990417,
                0.019843915477395058,
                0.017985843122005463,
                0.021959494799375534,
                0.02057054080069065,
                0.018678197637200356,
                0.020178476348519325,
                0.019597528502345085,
                0.022293532267212868,
                0.02360263466835022,
                0.018377292901277542,
                0.014294189400970936,
                0.018206333741545677,
                0.01955907605588436,
                0.024044692516326904,
                0.02063422091305256,
                0.022342098876833916,
                0.024129675701260567,
                0.015406979247927666,
                0.012902370654046535,
                0.013470296747982502,
                0.012218669056892395,
                0.01375508215278387,
                0.014444402419030666,
                0.012997922487556934,
                0.01066237036138773,
                0.018202347680926323,
                0.019554967060685158,
                0.013785217888653278,
                0.012707682326436043,
                0.015723273158073425,
                0.016791339963674545,
                0.013875293545424938,
                0.01527731865644455,
                0.015897350385785103,
                0.008886708877980709,
                0.00861489586532116,
                0.008715389296412468,
                0.008441098965704441,
                0.013297494500875473,
                0.0176148172467947,
                0.013111766427755356,
                0.006975357886403799,
                0.007544569205492735,
                0.011625256389379501,
                0.01148530188947916,
                0.009667829610407352,
                0.010215462185442448,
                0.00769251212477684,
                0.007555531337857246,
                0.00878791231662035,
                0.0097069526091218,
                0.011840583756566048,
                0.018135201185941696,
                0.015264364890754223,
                0.011252311989665031,
                0.008791887201368809,
                0.01029569748789072,
                0.011116103269159794,
                0.010752620175480843,
                0.011824819259345531,
                0.012925735674798489,
                0.013847263529896736,
                0.0154572743922472,
                0.013057484291493893,
                0.01240744348615408,
                0.010371671989560127,
                0.006789385806769133,
                0.00682962266728282,
                0.01036619208753109,
                0.014299118891358376,
                0.01378728449344635,
                0.011303319595754147,
                0.010440433397889137,
                0.008632753975689411,
                0.010186081752181053,
                0.01189603190869093,
                0.01259833574295044,
                0.012310040183365345,
                0.010757209733128548,
                0.00820740032941103,
                0.007701428607106209,
                0.011255553923547268,
                0.012210184708237648,
                0.010350912809371948,
                0.010199321433901787,
                0.008233514614403248,
                0.009990518912672997,
                0.015146956779062748,
                0.013698427006602287,
                0.009401137940585613,
                0.00813340861350298,
                0.006023115944117308,
                0.007605465594679117,
                0.013451828621327877,
                0.013628344982862473,
                0.010568490251898766,
                0.008371520787477493,
                0.007454524748027325,
                0.009787125512957573,
                0.011476760730147362,
                0.010218054987490177,
                0.007733708713203669,
                0.0167442187666893,
                0.04109583795070648,
                0.02088414691388607,
                0.01579882949590683,
                0.016957830637693405,
                0.015824362635612488,
                0.015263795852661133,
                0.012207789346575737,
                0.02322022244334221,
                0.023635931313037872,
                0.023629555478692055,
                0.03696505352854729,
                0.026286901906132698,
                0.03156459331512451,
                0.02386460267007351,
                0.021758589893579483,
                0.03501730412244797,
                0.03707921877503395,
                0.043815094977617264,
                0.08205108344554901,
                0.07777979224920273,
                0.06613963097333908,
                0.06977244466543198,
                0.04826647788286209,
                0.06758711487054825,
                0.039194002747535706,
                0.023872846737504005,
                0.04347619786858559,
                0.04712803661823273,
                0.038729455322027206,
                0.04454471170902252,
                0.05331975966691971,
                0.04424232989549637,
                0.032868754118680954,
                0.019055496901273727,
                0.010586035437881947,
                0.018142225220799446,
                0.056044112890958786,
                0.056771423667669296,
                0.03172220662236214,
                0.017641359940171242,
                0.03289493918418884,
                0.04526085779070854,
                0.038559067994356155,
                0.05887638032436371,
                0.040966130793094635,
                0.03403293713927269,
                0.07349403947591782,
                0.0662602037191391,
                0.04061330482363701,
                0.03975559026002884,
                0.046091362833976746,
                0.03828337788581848,
                0.02573205716907978,
                0.016194792464375496,
                0.021253734827041626,
                0.02427740953862667,
                0.018932389095425606,
                0.019722912460565567,
                0.03454894945025444,
                0.02306130714714527,
                0.04238971695303917,
                0.05642488971352577,
                0.11956103146076202,
                0.17318123579025269,
                0.1210353821516037,
                0.1244877278804779,
                0.14200620353221893,
                0.09141965955495834,
                0.06754018366336823,
                0.08520844578742981,
                0.11755970865488052,
                0.11343742907047272,
                0.08458554744720459,
                0.08274886012077332,
                0.13647690415382385,
                0.1112673208117485,
                0.0721137598156929,
                0.05228472501039505,
                0.04868456348776817,
                0.0508793368935585,
                0.057841673493385315,
                0.10058363527059555,
                0.14496350288391113
            ]
    # for e in x: 
    #     #if len(e) == 12:
    #     print(len(e))
    print(len(arr))

if __name__ == "__main__":

    preprocess_data_pcp(DATASET_PATH, JSON_PATH)
    # preprocess_data_pcp(DATA_TEST_PATH, JSON_PATH_TEST)
    # preprocess_data_cnn(DATASET_PATH, JSON_PATH)
    #length_check(data_path=JSON_PATH)