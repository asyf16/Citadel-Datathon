from datetime import datetime
import time
from data_loader import load_pair_data

def main():
    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Starting correlation analysis from {start_date} to {end_date}")
    tickers = ["MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BAX","BDX","BRK.B","BBY","BIIB","BLK","BX","BA","BKNG","BWA","BSX","BMY","AVGO","BRO","BF.B","BLDR","BG","BXP","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DAY","DECK","DE","DELL","DAL","DVN","DXCM","FANG","DLTR","D","DPZ","DOV","DOW","DHI","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","EG","EVRG","ES","EXC","EXPD","EXPE","EXR","XOM","FFIV","FDS","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FOX","FOXA","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GS","HAL","HIG","HAS","HCA","PEAK","HSIC","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TKO","TSCO","TTD","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VICI","V","VST","WAB","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR","WMB","WTW","WDAY","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS", "F","FAST","ALL","CARR","CAH","AME","D","TGT","OKE","IDXX","CMG","ADSK","VST","CBRE","PSA","MET","EA","AMP","FITB","DAL","CTVA","NDAQ","FANG","TER","CCL","HSY","ROK","EW","OXY","TRGP","DHI","YUM","XEL","EXC","COIN","NUE","ETR","FIX","WDAY","VMC","KR","ODFL","WAB","MLM","SYY","AIG","MCHP","MSCI","PEG","KEYS","RMD","HIG","DDOG","EBAY","HBAN","LVS","CPRT","IR","GRMN","VTR","ED","ROP","KDP","UAL","PYPL","CTSH","STT","GEHC","A","ACGL","MTB","WEC","TTWO","PCG","PRU","EL","EQT","PAYX","CCI","OTIS","KVUE","KMB","XYL","EME","XYZ","RJF","IBKR","NRG","FICO","AXON","LYV","DG","FISV","IQV","ADM","HPE","WTW","ROL","TPR","VICI","DOV","ULTA","TDY","EXR","STLD","BIIB","HAL","TSCO","CHTR","CFG","KHC","EXPE","ARES","CBOE","STZ","AEE","PPG","ATO","NTRS","IRM","LEN","LUV","MTD","DTE","DXCM","JBL","DVN","CINF","FE","FIS","RF","HUBB","PPL","WRB","WSM","PHM","EXE","ON","CNP","SYF","KEY","GIS","ES","TPL","VRSK","DRI","BRO","CPAY","STE","LDOS","EIX","DLTR","AVB","IP","EQR","AWK","CHD","EFX","CHRW","FSLR","HUM","CTRA","SW","L","LH","TSN","DOW","WAT","VLTO","BG","AMCR","CMS","EXPD","OMC","NVR","JBHT","PKG","PFG","CSGP","INCY","BR","DGX","NI","RL","VRSN","GPC","TROW","Q","SMCI","NTAP","GPN","LULU","DD","SBAC","ALB","SNA","WY","IFF","FTV","CNC","CDW","PTC","LII","MKC","HPQ","WST","ZBH","BALL","APTV","LYB","EVRG","J","LNT","PODD","TXT","VTRS","TKO","HOLX","ESS","DECK","NDSN","INVH","COO","MRNA","PNR","IEX","TRMB","FFIV","HII","MAA","ALLE","MAS","ERIE","TYL","GEN","AVY","KIM","BBY","CF","CLX","BEN","SWK","EG","BLDR","REG","FOX","HRL","AKAM","UHS","BF.B","SOLV","FOXA","ALGN","DPZ","HST","HAS","GDDY","TTD","ZBRA","JKHY","UDR","WYNN","AIZ","IVZ","DOC","SJM","GL","PSKY","RVTY","AES","CPT","IT","PNW","DAY","BAX","AOS","GNRC","NCLH","TECH","EPAM","BXP","MGM","TAP","POOL","APA","ARE","DVA","HSIC","CRL","SWKS","CAG","FRT","MOS","CPB","NWSA","FDS","MTCH","PAYC"]
]
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}")
        print(f"{'='*50}")

        try:
            df = load_pair_data(ticker, start_date, end_date, cache=True)
            if df.empty:
                print(f"No data for {ticker}")
                continue
        
        except Exception as e:
            print(f"Error processing {ticker}")
            continue

if __name__ == "__main__":
    main()
