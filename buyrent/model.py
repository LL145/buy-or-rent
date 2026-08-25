"""确定性核心: 买 vs 租 的终值财富比较。

约定:
- 现金流发生在每年年初(t=1..T), 期末 T 结算。
- 两条路径每年预算相同 = max(买方支出, 租方支出), 少花的一方把差额投入
  替代资产。这样比较的是"同样的钱走两条路"的期末净财富。
- 按揭为等额本息(年频)。利率在购买时锁定——这是"业主对冲通胀"的机制所在,
  蒙特卡洛中抽样的通胀只影响房价/租金/投资的名义路径, 不影响月供。
"""
from dataclasses import dataclass, replace


@dataclass
class Scenario:
    # ---- A层: 决策时刻可直接观察 ----
    price: float = 100.0        # 房价(归一化)
    rent_yield: float = 0.025   # 年租金 / 房价 (租售比的倒数)
    down: float = 0.30          # 首付比例
    mort_rate: float = 0.045    # 按揭名义年利率(锁定)
    mort_years: int = 30        # 按揭年限
    buy_cost: float = 0.03      # 购房交易成本(税费中介, 占房价)
    sell_cost: float = 0.04     # 卖房交易成本(占卖出价)
    carry: float = 0.012        # 年持有成本率(维护+物业+持有税费, 占当期房价)
    hold: int = 10              # 预期持有期 T(年)
    # ---- B/C层: 假设或抽样 ----
    g_house: float = 0.02       # 名义房价年涨幅
    g_rent: float = 0.02        # 名义租金年涨幅
    r_invest: float = 0.06      # 名义投资收益率(替代资产, 税后)
    infl: float = 0.02          # 通胀(用于换算真实值)

    def real(self, nominal: float) -> float:
        return (1 + nominal) / (1 + self.infl) - 1


def annuity_payment(principal: float, rate: float, years: int) -> float:
    if principal <= 0:
        return 0.0
    if rate == 0:
        return principal / years
    f = (1 + rate) ** years
    return principal * rate * f / (f - 1)


def loan_balance(principal: float, rate: float, years: int, after: int) -> float:
    """等额本息 after 年后的剩余本金。"""
    if principal <= 0 or after >= years:
        return 0.0
    pay = annuity_payment(principal, rate, years)
    if rate == 0:
        return principal - pay * after
    return principal * (1 + rate) ** after - pay * ((1 + rate) ** after - 1) / rate


def simulate(s: Scenario) -> dict:
    """返回两条路径的期末财富及分解。"""
    p0 = s.price
    loan = (1 - s.down) * p0
    pay = annuity_payment(loan, s.mort_rate, s.mort_years)

    # 年初现金流(t = 1..T); 租方把首付+交易成本先投出去
    buyer_side = 0.0
    renter_side = (s.down + s.buy_cost) * p0 * (1 + s.r_invest) ** s.hold
    for t in range(1, s.hold + 1):
        grow = (1 + s.r_invest) ** (s.hold - t + 1)   # 年初投入, 到期末的增值
        house_t = p0 * (1 + s.g_house) ** (t - 1)
        out_buy = (pay if t <= s.mort_years else 0.0) + s.carry * house_t
        out_rent = s.rent_yield * p0 * (1 + s.g_rent) ** (t - 1)
        budget = max(out_buy, out_rent)
        buyer_side += (budget - out_buy) * grow
        renter_side += (budget - out_rent) * grow

    house_T = p0 * (1 + s.g_house) ** s.hold
    balance_T = loan_balance(loan, s.mort_rate, s.mort_years, s.hold)
    wealth_buy = house_T * (1 - s.sell_cost) - balance_T + buyer_side
    wealth_rent = renter_side

    deflator = (1 + s.infl) ** s.hold
    return {
        "wealth_buy": wealth_buy,
        "wealth_rent": wealth_rent,
        "gap": wealth_buy - wealth_rent,               # >0 买划算
        "gap_pct_of_price": (wealth_buy - wealth_rent) / p0,
        "gap_real": (wealth_buy - wealth_rent) / deflator,
        "house_T": house_T,
        "loan_balance_T": balance_T,
        "buyer_side": buyer_side,
        "renter_side": renter_side,
    }


def with_growth(s: Scenario, g_house: float) -> Scenario:
    return replace(s, g_house=g_house)
