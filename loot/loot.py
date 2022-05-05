# Copyright (c) 2022 Andy Kee

import datetime

import pandas as pd
import numpy as np
import numpy_financial as npf
from dateutil.relativedelta import relativedelta

from loot.rates import USTAX, CATAX, STDED, IBND

# Reworked interfaces for useful numpy-financial functions
def pmt(rate, nper, prin):
    """Compute monthly loan payment (principal + interest)

    Parameters
    ----------
    rate : float
        Interest rate (APR)
    nper : int
        Loan term (months)
    prin : float
        Loan principal

    Returns
    -------
    pmt : float
        Monthly payment

    """
    return -npf.pmt(rate=rate/12, nper=nper, pv=prin)


def ppmt(rate, per, nper, prin):
    """Compute monthly payment against loan principal
    
    Parameters
    ----------
    rate : float
        Interest rate (APR)
    per : int
        Current period
    nper : int
        Loan term (months)
    prin : float
        Loan principal

    Returns
    -------
    pmt : float
        Monthly principal payment

    """
    return -npf.ppmt(rate=rate/12, per=per, nper=nper, pv=prin)


def ipmt(rate, per, nper, prin):
    """Compute monthly interest payment
    
    Parameters
    ----------
    rate : float
        Interest rate (APR)
    per : int
        Current period
    nper : int
        Loan term (months)
    prin : float
        Loan principal

    Returns
    -------
    pmt : float
        Monthly interest payment

    """
    return -npf.ipmt(rate=rate/12, per=per, nper=nper, pv=prin)


def fv(rate, nper, prin, pmt=0):
    """Compute future value

    Parameters
    ----------
    rate : float
        Interest rate (APY)
    nper : int
        Number of compounding months
    prin : float
        Present value
    pmt : float, optional
        Additional monthly principal

    Returns
    -------
    fv : float
        Future value

    """
    return npf.fv(rate=rate/12, nper=nper, pv=-prin, pmt=-pmt)


def bal(rate, per, nper, prin):
    """Compute loan balance

    Parameters
    ----------
    rate : float
        Interest rate (APR)
    per : int
        Current period
    nper : int
        Loan term (months)
    prin : float
        Loan principal

    Returns
    -------
    pmt : float
        Current loan balance

    """
    l = ldf(rate, nper, prin)
    return l.at[per, 'Balance']


def ldf(rate, nper, prin, round=2, start_date=None):
    """Create a loan dataframe

    """
    df = pd.DataFrame(index=np.arange(1, nper+1), columns=['Payment', 'Principal', 'Interest', 'Balance'])
    df.index.name = 'Month'

    if start_date:
        start = pd.date_range(start=start_date, end=start_date, periods=1)
        year, month, day = start[0].year, start[0].month, start[0].day
        offset = day - 1

        ts = pd.date_range(start=f'{month}/1/{year}', periods=nper, freq='MS')
        ts += (day-1) * pd.offsets.Day()
        df.insert(0, 'Date', ts)

    df['Payment'] = pmt(rate=rate, nper=nper, prin=prin)
    df['Principal'] = ppmt(rate=rate, per=df.index, nper=nper, prin=prin)
    df['Interest'] = ipmt(rate=rate, per=df.index, nper=nper, prin=prin)
    df['Balance'] = prin - df['Principal'].cumsum()
    df = df.round(round)
    
    return df


def itax(tinc, table):
    """Compute income tax

    Parameters
    ----------
    tinc : float
        Taxable income
    table: dict
        Tax table 

    Returns
    -------
    itax : float
        Income tax owed

    """
    tax = 0
    brackets = list(table.keys())
    for n, bracket in enumerate(brackets):
        if tinc > brackets[n+1]:
            tax += (brackets[n+1] - brackets[n]) * table[bracket]
        else:
            tax += (tinc - brackets[n]) * table[bracket]
            break
    return round(tax, 2)


def ustax(tinc, fstatus='MFJ', year=None):
    """Compute federal income tax

    Parameters
    ----------
    tinc : float
        Taxable income
    fstatus: str
        Filing status. Current support for married filing jointly
        ('MFJ')

    Returns
    -------
    itax : float
        Income tax owed

    """
    if year is None:
        year = datetime.date.today().year
    return itax(tinc, USTAX[year][fstatus]) 


def catax(tinc, fstatus='MFJ', year=None):
    """Compute California income tax

    Parameters
    ----------
    tinc : float
        Taxable income
    fstatus: str
        Filing status. Current support for married filing jointly
        ('MFJ')

    Returns
    -------
    itax : float
        Income tax owed

    """
    if year is None:
        year = datetime.date.today().year
    return itax(tinc, CATAX[year][fstatus]) 


def dedmint(rate, nper, prin, round=2):
    prin = 750000 if prin > 750000 else prin
    df = ldf(rate=rate, nper=nper, prin=prin, round=round)
    
    mint = []
    for year in _chunker(df, 12):
        mint.append(year['Interest'].sum())

    df = pd.DataFrame(index=np.arange(1, len(mint)+1), columns=['Interest'])
    df.index.name = 'Year'
    df['Interest'] = mint
    df = df.round(round)
    return df

    
def _chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def hoins(val, pct=0.0025):
    """Estimate annual homeowner's insurance payment

    Parameters
    ----------
    val : float
        Home value
    pct : float, optional
       Percentage of home value charged. Default is 0.0025

    Returns
    -------
    hins : float
        Estimated annual cost

    """
    return val * pct


def ptax(val, rate=0.0127):
    """Compute annual property tax payment

    Parameters
    ----------
    val : float
        Taxable home value
    rate : float
        Property tax rate

    Returns
    -------
    ptax : float
        Annual property tax

    """
    return val * rate


def stded(fstatus='MFJ', year=None):
    """Return federal standard deduction
    
    """
    if year is None:
        year = datetime.date.today().year
    return STDED[year][fstatus]


def ibnd(pdate, prin, date=None, early_penalty=True):
    """Compute US Treasury I bond value
    
    Parameters
    ----------
    pdate : str
        Purchase date formatted as 'YYYY-MM-DD'
    prin : int
        Original bond purchase
    date : str
        Date to evaluate the bond value at, formatted as 'YY-MM-DD'.
        If None (default), the current date is used. Note that date
        must be in the past.
    early_penalty : bool
        If True (default), the returned value is reduced by last
        three months interest if bond is held less than 5 years.

    Returns
    -------
    val : int
        Present bond value

    Notes
    -----
    * Bonds redemption value is rounded to the nearest $4
    * Bonds held less than 5 years forefit the last 3 months of interest

    """
    prin = [prin]
    pdate = datetime.date.fromisoformat(pdate)
    rdate = _rate_date(pdate)
    fixed_rate = IBND[rdate][0]

    # redemption date (really month)
    redate = datetime.date.today() if date is None else datetime.date.fromisoformat(date)
    redate.replace(day=1)
    
  

    date = pdate
    m = 1

    while date < redate:
        floating_rate = IBND[rdate][1]
        composite_rate = fixed_rate + (2*floating_rate) + (fixed_rate * floating_rate)
        prin.append(fv(composite_rate, 1, prin[-1]))
        date += relativedelta(months=1)
        m += 1

        if m > 6:
            rdate = _rate_date(date)
            m = 1    

    if early_penalty:
        # Forefit last 3 months of interest if held less than 5 years
        val = prin[-1] if len(prin) > 60 else prin[-4]
    else:
        val = prin[-1]

    # Bonds are rounded to the nearest $4 when redeemed
    rem = val % 4
    val = val - rem if rem < 2 else val - rem + 4
    return val


def _rate_date(pdate):
    rate_year = pdate.year
    if pdate.month < 5:
        rate_month = 11
        rate_year -= 1
    elif pdate.month < 11:
        rate_month = 5
    else:
        rate_month = 11
    return datetime.date(rate_year, rate_month, 1).strftime('%Y-%m-%d')
