*******************************************************
* do-file: nl_did_vpyield_fixed.do
* 目的  : 导入 csv -> 类型转换 -> 筛样 -> 用 nl 估计 Diebold-Li 式模型
*         在每次 nl 迭代中剖出 company & month 固定效应
* 重要修正: 在 nl 表达式中使用 {L}，起始值通过 initial(L = `startL') 传入
*******************************************************

capture log close
cd "D:\Projects\python\jupyter\final_pj"

* -------------------------
* 1) 导入 CSV
* -------------------------
import delimited using "bond_return_with_kcbz.csv", ///
    varnames(1) case(preserve) clear

* -------------------------
* 2) 字段类型转换
* -------------------------
foreach d in EndDt ValueDt MatDt {
    capture confirm variable `d'
    if _rc == 0 {
        capture confirm string variable `d'
        if _rc == 0 {
            gen double `d'_td = date(trim(`d'), "YMD") if trim(`d') != ""
            format `d'_td %td
            drop `d'
            rename `d'_td `d'
        }
        else {
            format `d' %td
        }
    }
    else {
        display as text "Note: variable `d' not found; skipping conversion for `d'."
    }
}

foreach v in VPYield Maturity TrueYield {
    capture confirm variable `v'
    if _rc == 0 {
        capture confirm string variable `v'
        if _rc == 0 {
            gen double `v'_num = real(trim(`v')) if trim(`v') != ""
            drop `v'
            rename `v'_num `v'
        }
        else {
            recast double `v'
        }
    }
    else {
        display as text "Note: variable `v' not found; please check your CSV."
    }
}

* kcbz -> dummy
capture confirm numeric variable kcbz
if _rc == 0 {
    gen byte kcbz_num = (kcbz != 0) if !missing(kcbz)
}
else {
    capture confirm string variable kcbz
    if _rc == 0 {
        gen byte kcbz_num = .
        replace kcbz_num = 1 if inlist(lower(trim(kcbz)), "1", "y", "yes", "true", "t")
        replace kcbz_num = 0 if inlist(lower(trim(kcbz)), "0", "n", "no", "false", "f")
        replace kcbz_num = real(trim(kcbz)) if missing(kcbz_num) & regexm(trim(kcbz), "^[0-9]+$")
        replace kcbz_num = 0 if missing(kcbz_num)
    }
    else {
        display as error "ERROR: variable kcbz not found; creating Treated = 0."
        gen byte kcbz_num = 0
    }
}
label variable kcbz_num "kcbz as numeric dummy"
capture confirm variable kcbz
if _rc == 0 drop kcbz
rename kcbz_num kcbz
gen byte Treated = kcbz
label variable Treated "处理组 (kcbz dummy)"

* -------------------------
* 3) 检查 CompanyCode 与 EndDt，并生成 EndDt_month
* -------------------------
capture confirm variable CompanyCode
if _rc != 0 {
    display as error "ERROR: CompanyCode not found. Please ensure a company identifier exists."
    exit 198
}

capture confirm variable EndDt
if _rc != 0 {
    capture confirm variable ValueDt
    if _rc == 0 {
        rename ValueDt EndDt
    }
    else {
        capture confirm variable IssDt
        if _rc == 0 {
            rename IssDt EndDt
        }
        else {
            display as error "ERROR: No EndDt/ValueDt/IssDt found. Please provide a date column."
            exit 198
        }
    }
}

gen int EndDt_month = year(EndDt)*100 + month(EndDt)

* -------------------------
* 4) 过滤 BdType（保留 13,14,19）
* -------------------------
capture confirm numeric variable BdType
if _rc == 0 {
    keep if inlist(BdType, 13, 14, 19)
}
else {
    capture confirm string variable BdType
    if _rc == 0 {
        tempvar BdType_num
        gen double `BdType_num' = real(trim(BdType)) if !missing(BdType)
        keep if inlist(`BdType_num', 13, 14, 19)
        drop `BdType_num'
    }
    else {
        display as error "ERROR: BdType not found."
        exit 198
    }
}

* -------------------------
* 5) 时间窗口、Post、DiD
* -------------------------
local key_date = td(18jun2025)
gen byte time_window = (abs(EndDt - `key_date') <= 90)
keep if time_window == 1

gen byte Post = (EndDt >= `key_date')
label variable Post "政策后时期"

gen byte DiD = Treated * Post
label variable DiD "处理组×政策后"

* -------------------------
* 6) 描述统计
* -------------------------
summarize VPYield Maturity Post Treated DiD TrueYield

* -------------------------
* 7) 定义 function-evaluator 程序 （必须在 nl 调用之前定义）
* -------------------------
capture program drop nl_dld_fe
program define nl_dld_fe, rclass
    version 17.0
    args lnf b1Cb b2Cb b3Cb b1Ca b2Ca b3Ca b1Tb b2Tb b3Tb b1Ta b2Ta b3Ta L

    tempvar touse term1 term2 f_it resid alpha gamma alpha_old gamma_old tmp tmp2 modelval ///
        cmean mmean cmean2 chg
    gen byte `touse' = !missing(VPYield, Maturity, Post, Treated, DiD)

    gen double `term1' = .
    replace `term1' = ((1 - exp(-`L'*Maturity)) / (`L'*Maturity)) ///
        if `touse' & `L'!=0 & Maturity>0
    replace `term1' = 1 if `touse' & (`L'==0 | Maturity==0)

    gen double `term2' = .
    replace `term2' = (`term1' - exp(-`L'*Maturity)) if `touse'

    gen double `f_it' = ///
        (`b1Cb') + (`b2Cb')*`term1' + (`b3Cb')*`term2' + ///
        (`b1Ca')*Post + (`b2Ca')*Post*`term1' + (`b3Ca')*Post*`term2' + ///
        (`b1Tb')*Treated + (`b2Tb')*Treated*`term1' + (`b3Tb')*Treated*`term2' + ///
        (`b1Ta')*DiD + (`b2Ta')*DiD*`term1' + (`b3Ta')*DiD*`term2'

    gen double `resid' = VPYield - `f_it' if `touse'
    gen double `alpha' = 0 if `touse'
    gen double `gamma' = 0 if `touse'
    bysort CompanyCode: egen double `cmean' = mean(`resid') if `touse'
    replace `alpha' = `cmean' if `touse'
    drop `cmean'

    gen double `alpha_old' = `alpha' if `touse'
    gen double `gamma_old' = `gamma' if `touse'

    local maxiter = 50
    local tol = 1e-10
    forvalues iter = 1/`maxiter' {
        replace `alpha_old' = `alpha' if `touse'
        replace `gamma_old' = `gamma' if `touse'

        gen double `tmp' = `resid' - `alpha' if `touse'
        bysort EndDt_month: egen double `mmean' = mean(`tmp') if `touse'
        replace `gamma' = `mmean' if `touse'
        drop `tmp' `mmean'

        gen double `tmp2' = `resid' - `gamma' if `touse'
        bysort CompanyCode: egen double `cmean2' = mean(`tmp2') if `touse'
        replace `alpha' = `cmean2' if `touse'
        drop `tmp2' `cmean2'

        quietly summarize `alpha' if `touse', meanonly
        local a_bar = r(mean)
        quietly summarize `gamma' if `touse', meanonly
        local g_bar = r(mean)
        local center = (`a_bar' + `g_bar')
        replace `alpha' = `alpha' - `center'/2 if `touse'
        replace `gamma' = `gamma' - `center'/2 if `touse'

        gen double `chg' = abs((`alpha' + `gamma') - (`alpha_old' + `gamma_old')) if `touse'
        quietly summarize `chg' if `touse', meanonly
        local maxchg = r(max)
        drop `chg'

        if `maxchg' < `tol' {
            continue, break
        }
    }

    gen double `modelval' = `f_it' + `alpha' + `gamma' if `touse'
    replace `lnf' = VPYield - `modelval' if `touse'
    replace `lnf' = . if !`touse'

    return scalar nl_fe_altern_iters = `iter'
end

* 检查程序确实在内存
program list nl_dld_fe
* 如果输出代码则说明程序已成功定义，可继续下一步。
* 如果显示 "nl_dld_fe not found" 请先运行上面的 program define 块，确认无语法错误。

* -------------------------
* 8) 调用 nl（custom evaluator + FE）
* -------------------------
nl nl_dld_fe @ VPYield Maturity Post Treated DiD, ///
    parameters(b1Cb b2Cb b3Cb b1Ca b2Ca b3Ca b1Tb b2Tb b3Tb b1Ta b2Ta b3Ta L) ///
    iterate(10000) ///
    initial(b1Cb = .01 b2Cb = .01 b3Cb = .01 b1Ca = 0 b2Ca = 0 b3Ca = 0 ///
            b1Tb = 0 b2Tb = 0 b3Tb = 0 b1Ta = 0 b2Ta = 0 b3Ta = 0 L = 1) ///
    vce(cluster CompanyCode)

estimates store nl_regression

* -------------------------
* 9) 输出、预测、绘图、边际效应
* -------------------------
estimates table nl_regression, star(0.1 0.05 0.01) stats(N r2) b(%9.4f)
predict y_hat

twoway scatter VPYield y_hat if e(sample), colorvar(BdType) colordiscrete ///
    title("实际值 vs 预测值 (按债券类型着色)") ///
    xtitle("预测值") ///
    ytitle("实际发行利率 (VPYield)")

graph export "scatter_by_bdtype.png", replace

margins, dydx(DiD) at(Maturity = (0.25 0.5 1 2 3 5 10 15))

save "did_analysis_data2.dta", replace

*******************************************************
* End of do-file
*******************************************************
