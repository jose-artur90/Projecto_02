import sys
import matplotlib
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

df = pandas.read_csv("hotel_booking.csv")

d = {'Resort Hotel': 0, 'City Hotel': 1}
df['hotel'] = df['hotel'].map(d)

d = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11}
df['arrival_date_month'] = df['arrival_date_month'].map(d)

d = {'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4}
df['meal'] = df['meal'].map(d)

d = {'ABW': 1, 'AGO': 2, 'AIA': 3, 'ALB' : 4, 'AND': 5, 'ARE': 6, 'ARG': 7, 'ARM': 8, 'ASM': 9, 'ATA': 10, 'ATF': 11, 'AUT': 12, 'AUS': 13,
    'AZE': 14, 'BDI': 15, 'BEL': 16, 'BEN': 17, 'BGD': 18, 'BGR': 19, 'BHR': 20, 'BHS': 21, 'BIH': 22, 'BLR': 23, 'BOL': 24, 'BRA': 25, 
    'BRB': 26, 'BWA': 27, 'CAF': 28, 'CHE': 29, 'CHL': 30, 'CHN': 31, 'CIV': 32, 'CMR':33, 'CN':34, 'COL': 35, 'COM':36, 'CPV':37, 'CRI':38, 
    'CUB':39, 'CYM':40, 'CYP':41, 'CZE':42, 'DEU':43, 'DJI':44, 'DMA':45, 'DNK':46, 'DOM':47, 'DZA':48, 'ECU':49, 'EGY':50, 'ESP':51, 'EST':52,
    'ETH':53, 'FIN':54, 'FJI':55, 'FRA':56, 'FRO':57, 'GAB':58, 'GBR':59, 'GEO':60, 'GGY':61, 'GHA':62, 'GIB':63, 'GLP':64, 'GNB':65, 'GRC':66, 
    'GTM':67, 'GUY':68, 'HKG':69, 'HND':70, 'HRV':71, 'HUN':72, 'IDN':73, 'IMN':74, 'IND':75, 'IRL':76, 'IRN':77, 'IRQ':78, 'ISL':79, 'ISR':80, 
    'ITA':81, 'JAM':82, 'JEY':83, 'JOR':84, 'JPN':85, 'KAZ':86,  'KEN':87, 'KHM':88, 'KIR':89, 'KNA':90, 'KOR':91, 'KWT':92, 'LAO':93, 'LBN':94, 
    'LBY':95, 'LCA':96, 'LIE':97, 'LKA':98, 'LTU':99, 'LUX':100, 'LVA':0, 'MAC':101, 'MAR':102, 'MCO':103, 'MDG':104, 'MDV':105, 'MEX':106, 
    'MKD':107, 'MLI':108, 'MLT':109, 'MMR':110, 'MNE':111, 'MOZ':112, 'MRT':113, 'MUS':114, 'MWI':115, 'MYS':116, 'MYT':117, 'NAM':118, 
    'NCL':119, 'NGA':120, 'NIC':121, 'NLD':122, 'NOR':123, 'NPL':124, 'NZL':125, 'OMN':126, 'PAK':127, 'PAN':128, 'PER':129, 'PHL':130, 
    'PLW':131, 'POL':132, 'PRI':133, 'PRT':134, 'PRY':135, 'PYF':136, 'QAT':137, 'ROU':138, 'RUS':139, 'RWA':140, 'SAU':141, 'SDN':142, 
    'SEN':143, 'SGP':144, 'SLV':145, 'SMR':146, 'SRB':147, 'STP':148, 'SUR':149, 'SVK':150, 'SVN':151, 'SWE':152, 'SYC':153, 'SYR':154, 
    'TGO':155, 'THA':156, 'TJK':157, 'TMP':158, 'TUN':159, 'TUR':160, 'TWN':161, 'TZA':162, 'UGA':163, 'UKR':164, 'UMI':165, 'URY':166, 
    'USA':167, 'UZB':168, 'VEN':169, 'VGB':170,'VNM':171, 'ZAF':172, 'ZMB':173, 'ZME' :174, '':175}
df['country'] = df['country'].map(d)

d = {'Aviation': 0, 'Complementary': 1, 'Corporate': 2, 'Direct': 3, 'Undefined': 4, 'Groups':5, 'Offline TA/TO': 6, 'Online TA': 7}
df['market_segment'] = df['market_segment'].map(d)

d = {'Canceled': 1, 'Check-Out': 0}
df['reservation_status'] = df['reservation_status'].map(d)

features = ['hotel','is_canceled','lead_time','arrival_date_year','stays_in_weekend_nights','stays_in_week_nights','adults',
            'children','babies','meal', 'market_segment','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled',
            'booking_changes','days_in_waiting_list','adr','required_car_parking_spaces','total_of_special_requests', 'reservation_status']

X = df[features]
y = df['reservation_status']


#Preencher valores ausentes com valores arbitrários
X[np.isnan(X)] = -1
y[np.isnan(y)] = -1

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# tree.plot_tree(dtree, feature_names=features)

# Para guardar a árvore de decisão como Imagem
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtree, feature_names=features, filled=True)        
fig.savefig("decistion_tree.png")

print(fig)

#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()



