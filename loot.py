# Copyright (c) 2022 Andy Kee

import pandas as pd
import numpy as np
import numpy_financial as npf

__version__ = "1.0.0"

# 2022 federal standard deduction
STDED = {'MFJ': 25100}

# 2021 federal tax table
USTAX = {'MFJ' : {0: 0.1, 
                  19900: 0.12,
                  81050: 0.22,
                  172750: 0.24,
                  329850: 0.32,
                  418850: 0.35,
                  628300: 0.37}}

# 2021 california tax table
CATAX = {'MFJ' : {0: 0.01,
                  18650: 0.02,
                  44214: 0.04,
                  69784: 0.06,
                  96870: 0.08,
                  122428: 0.093,
                  625372: 0.103,
                  750442: 0.113,
                  1250738: 0.123}}

IBND = {'2021-05-01': (0, 0.0177),
        '2021-11-01': (0, 0.0356)}

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


def ldf(rate, nper, prin, round=2):
    """Create a loan dataframe

    """
    df = pd.DataFrame(index=np.arange(1, nper+1), columns=['Payment', 'Principal', 'Interest', 'Balance'])
    df.index.name = 'Month'

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


def ustax(tinc, fstatus):
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
    return itax(tinc, USTAX[fstatus]) 


def catax(tinc, fstatus):
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
    return itax(tinc, CATAX[fstatus]) 


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


def stded(fstatus='MFJ'):
    """Return federal standard deduction
    
    """
    return STDED[fstatus]


def ibnd(pdate, prin):
    date = datetime.date.fromisoformat(pdate)
    rate_date = _rate_date(date)
    fixed_rate = IBND[rate_date][0]
    m = 1

    while date < datetime.date.today().replace(day=1):
        floating_rate = IBND[rate_date][1]
        composite_rate = fixed_rate + (2*floating_rate) + (fixed_rate * floating_rate)
        if m == 6:
            rate_date = _rate_date(date)
            m = 1
    
        prin = loot.fv(composite_rate, 1, prin)
        date += relativedelta(months=1)
        m += 1

    return prin


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
