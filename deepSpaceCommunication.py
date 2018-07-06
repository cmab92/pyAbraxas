import numpy as np
import matplotlib.pyplot as plt

d = 300                                     # Durchmesser der Antenne(n) in m
P = 66                                      # Sendeleistung in dBW ( X_deciBel = 10*log10(x_linear/1W) ), 60dBW -> 1MW, 66dBW -> 4MW
f = 10*10**9                                 # Sendefrequenz (vgl. FAST, steigt die Frequenz bei konstanter Antennengröße verbessert sich die Übertragung nur da der Antennengewinn steigt)
R = np.linspace(0.5, 10, 100)               # Distanz in Lichtjahren (die nächstgelegene (2,5 Mio. LJ) Galaxie ist die Andromedagalaxie)

G = 10*np.log10((np.pi*d/(299792458/f))**2*0.9)                         # geschätzter Antennengewinn in dBi, also im Verhältnis zum (isotropen) Kugelstrahler
print("Antennengewinn: ", G, " dBi")
theta = 58.8*299792458/f*1/d                                            # Näherung der 3dB-Breite der Hauptkeule
print("Halbwertsbreite der Antenne: ", theta, "°")

P_r = P + 2*G + 20*np.log10((299792458/f)/(4*np.pi*(R*9.461*10**15)))   # Empfangsleistung (Friis-Gleichung), baugleiche Sende- und Empfangsantenne



for i in range(2):
    S = 10**i                                   # Größe des Datenpakets in GB
    D = 8*S*10**9                               # Transinformation in bit
    B = 2*10**9                                 # 2 GHz Bandbreite (eigentlich nicht so plausibel, tatsächlich wohl wesentlich geringer)
    T = 290                                     # Rauschtemperatur in K
    k = 1.38064852*10**(-23)                    # Ws/K
    N = k*B*T                                   # Rauschleistung am Empfänger
    C = B*np.log2(1 + (10**(P_r/10))/N)/10**6   # Kanalkapazität in Mb/s
    tau = D/C/60/60/24
    plt.plot(R, tau/60/60, label=" Übertragungsdauer in Tagen (Datenvolumen: " + str(int(S)) + "GB)")

# plots
plt.plot(R, C, label="Shannon-Kanalkapazität")
plt.legend()
plt.xlabel("Distanz in Lichtjahren")
plt.ylim([-2.5, 20])
plt.ylabel("Datenrate in Mb/s (Kanalkapazität) oder Zeit in Tagen (Übertragungsdauer)")
plt.title("Bandbreite B = " + str(int(B/10**6)) + " MHz, Sendeleistung = " + str(P) + " dBW")
plt.grid()
plt.show()

# ohne Gewähr :)

