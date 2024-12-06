import itertools
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from udao_trace.utils import JsonHandler


def get_table_info() -> Tuple[Dict[str, Dict], List[str]]:
    # using all the numerical columns from the TPC-DS schema (like JOB)
    tables: Dict[str, Dict] = {
        "call_center": {
            "alias": "cc",
            "columns": {
                "cc_call_center_sk": {"max": 30, "min": 1},
                "cc_company": {"max": 6, "min": 1},
                "cc_division": {"max": 6, "min": 1},
                "cc_employees": {"max": 69113, "min": 3180},
                "cc_gmt_offset": {"max": -5.0, "min": -6.0},
                "cc_mkt_id": {"max": 6, "min": 1},
                "cc_open_date_sk": {"max": 2451120, "min": 2450812},
                # "cc_rec_end_date": {"max": "2001-12-31", "min": "2000-01-01"},
                # "cc_rec_start_date": {"max": "2002-01-01", "min": "1998-01-01"},
                "cc_sq_ft": {"max": 43081086, "min": 531060},
                "cc_tax_percentage": {"max": 0.12, "min": 0.0},
            },
            "foreign_keys": [],
        },
        "catalog_page": {
            "alias": "cp",
            "columns": {
                "cp_catalog_number": {"max": 109, "min": 1},
                "cp_catalog_page_number": {"max": 188, "min": 1},
                "cp_catalog_page_sk": {"max": 20400, "min": 1},
                "cp_end_date_sk": {"max": 2453186, "min": 2450844},
                "cp_start_date_sk": {"max": 2453005, "min": 2450815},
            },
            "foreign_keys": [],
        },
        "catalog_returns": {
            "alias": "cr",
            "columns": {
                "cr_call_center_sk": {"max": 30, "min": 1},
                "cr_catalog_page_sk": {"max": 17108, "min": 1},
                "cr_fee": {"max": 100.0, "min": 0.5},
                "cr_item_sk": {"max": 204000, "min": 1},
                "cr_net_loss": {"max": 16033.97, "min": 0.5},
                "cr_order_number": {"max": 15999998, "min": 3},
                "cr_reason_sk": {"max": 55, "min": 1},
                "cr_refunded_addr_sk": {"max": 1000000, "min": 1},
                "cr_refunded_cash": {"max": 25886.39, "min": 0.0},
                "cr_refunded_cdemo_sk": {"max": 1920800, "min": 1},
                "cr_refunded_customer_sk": {"max": 2000000, "min": 1},
                "cr_refunded_hdemo_sk": {"max": 7200, "min": 1},
                "cr_return_amount": {"max": 27559.64, "min": 0.0},
                "cr_return_amt_inc_tax": {"max": 29529.57, "min": 0.0},
                "cr_return_quantity": {"max": 100, "min": 1},
                "cr_return_ship_cost": {"max": 14159.97, "min": 0.0},
                "cr_return_tax": {"max": 2438.22, "min": 0.0},
                "cr_returned_date_sk": {"max": 2452923, "min": 2450822},
                "cr_returned_time_sk": {"max": 86399, "min": 0},
                "cr_returning_addr_sk": {"max": 1000000, "min": 1},
                "cr_returning_cdemo_sk": {"max": 1920800, "min": 1},
                "cr_returning_customer_sk": {"max": 2000000, "min": 1},
                "cr_returning_hdemo_sk": {"max": 7200, "min": 1},
                "cr_reversed_charge": {"max": 22506.38, "min": 0.0},
                "cr_ship_mode_sk": {"max": 20, "min": 1},
                "cr_store_credit": {"max": 20901.46, "min": 0.0},
                "cr_warehouse_sk": {"max": 15, "min": 1},
            },
            "foreign_keys": [
                {
                    "column": "cr_returned_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "cr_returned_time_sk",
                    "ref_table": "time_dim",
                    "ref_column": "t_time_sk",
                },
                {
                    "column": "cr_item_sk",
                    "ref_table": "item",
                    "ref_column": "i_item_sk",
                },
                {
                    "column": "cr_item_sk",
                    "ref_table": "catalog_sales",
                    "ref_column": "cs_item_sk",
                },
                {
                    "column": "cr_refunded_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "cr_refunded_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "cr_refunded_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "cr_refunded_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "cr_returning_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "cr_returning_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "cr_returning_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "cr_returning_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "cr_call_center_sk",
                    "ref_table": "call_center",
                    "ref_column": "cc_call_center_sk",
                },
                {
                    "column": "cr_catalog_page_sk",
                    "ref_table": "catalog_page",
                    "ref_column": "cp_catalog_page_sk",
                },
                {
                    "column": "cr_ship_mode_sk",
                    "ref_table": "ship_mode",
                    "ref_column": "sm_ship_mode_sk",
                },
                {
                    "column": "cr_warehouse_sk",
                    "ref_table": "warehouse",
                    "ref_column": "w_warehouse_sk",
                },
                {
                    "column": "cr_reason_sk",
                    "ref_table": "reason",
                    "ref_column": "r_reason_sk",
                },
                {
                    "column": "cr_order_number",
                    "ref_table": "catalog_sales",
                    "ref_column": "cs_order_number",
                },
            ],
        },
        "catalog_sales": {
            "alias": "cs",
            "columns": {
                "cs_bill_addr_sk": {"max": 1000000, "min": 1},
                "cs_bill_cdemo_sk": {"max": 1920800, "min": 1},
                "cs_bill_customer_sk": {"max": 2000000, "min": 1},
                "cs_bill_hdemo_sk": {"max": 7200, "min": 1},
                "cs_call_center_sk": {"max": 30, "min": 1},
                "cs_catalog_page_sk": {"max": 17108, "min": 1},
                "cs_coupon_amt": {"max": 28730.0, "min": 0.0},
                "cs_ext_discount_amt": {"max": 29904.0, "min": 0.0},
                "cs_ext_list_price": {"max": 29994.0, "min": 1.0},
                "cs_ext_sales_price": {"max": 29787.0, "min": 0.0},
                "cs_ext_ship_cost": {"max": 14992.0, "min": 0.0},
                "cs_ext_tax": {"max": 2612.83, "min": 0.0},
                "cs_ext_wholesale_cost": {"max": 10000.0, "min": 1.0},
                "cs_item_sk": {"max": 204000, "min": 1},
                "cs_list_price": {"max": 300.0, "min": 1.0},
                "cs_net_paid": {"max": 29787.0, "min": 0.0},
                "cs_net_paid_inc_ship": {"max": 43777.0, "min": 0.0},
                "cs_net_paid_inc_ship_tax": {"max": 46367.65, "min": 0.0},
                "cs_net_paid_inc_tax": {"max": 31644.35, "min": 0.0},
                "cs_net_profit": {"max": 19858.0, "min": -10000.0},
                "cs_order_number": {"max": 16000000, "min": 1},
                "cs_promo_sk": {"max": 1000, "min": 1},
                "cs_quantity": {"max": 100, "min": 1},
                "cs_sales_price": {"max": 300.0, "min": 0.0},
                "cs_ship_addr_sk": {"max": 1000000, "min": 1},
                "cs_ship_cdemo_sk": {"max": 1920800, "min": 1},
                "cs_ship_customer_sk": {"max": 2000000, "min": 1},
                "cs_ship_date_sk": {"max": 2452744, "min": 2450817},
                "cs_ship_hdemo_sk": {"max": 7200, "min": 1},
                "cs_ship_mode_sk": {"max": 20, "min": 1},
                "cs_sold_date_sk": {"max": 2452654, "min": 2450815},
                "cs_sold_time_sk": {"max": 86399, "min": 0},
                "cs_warehouse_sk": {"max": 15, "min": 1},
                "cs_wholesale_cost": {"max": 100.0, "min": 1.0},
            },
            "foreign_keys": [
                {
                    "column": "cs_sold_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "cs_sold_time_sk",
                    "ref_table": "time_dim",
                    "ref_column": "t_time_sk",
                },
                {
                    "column": "cs_ship_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "cs_bill_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "cs_bill_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "cs_bill_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "cs_bill_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "cs_ship_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "cs_ship_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "cs_ship_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "cs_ship_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "cs_call_center_sk",
                    "ref_table": "call_center",
                    "ref_column": "cc_call_center_sk",
                },
                {
                    "column": "cs_catalog_page_sk",
                    "ref_table": "catalog_page",
                    "ref_column": "cp_catalog_page_sk",
                },
                {
                    "column": "cs_ship_mode_sk",
                    "ref_table": "ship_mode",
                    "ref_column": "sm_ship_mode_sk",
                },
                {
                    "column": "cs_warehouse_sk",
                    "ref_table": "warehouse",
                    "ref_column": "w_warehouse_sk",
                },
                {
                    "column": "cs_item_sk",
                    "ref_table": "item",
                    "ref_column": "i_item_sk",
                },
                {
                    "column": "cs_promo_sk",
                    "ref_table": "promotion",
                    "ref_column": "p_promo_sk",
                },
            ],
        },
        "customer": {
            "alias": "c",
            "columns": {
                "c_birth_day": {"max": 31, "min": 1},
                "c_birth_month": {"max": 12, "min": 1},
                "c_birth_year": {"max": 1992, "min": 1924},
                "c_current_addr_sk": {"max": 999999, "min": 1},
                "c_current_cdemo_sk": {"max": 1920799, "min": 1},
                "c_current_hdemo_sk": {"max": 7200, "min": 1},
                "c_customer_sk": {"max": 2000000, "min": 1},
                "c_first_sales_date_sk": {"max": 2452648, "min": 2448998},
                "c_first_shipto_date_sk": {"max": 2452678, "min": 2449028},
            },
            "foreign_keys": [],
        },
        "customer_address": {
            "alias": "ca",
            "columns": {
                "ca_address_sk": {"max": 1000000, "min": 1},
                "ca_gmt_offset": {"max": -5.0, "min": -10.0},
            },
            "foreign_keys": [],
        },
        "customer_demographics": {
            "alias": "cd",
            "columns": {
                "cd_demo_sk": {"max": 1920800, "min": 1},
                "cd_dep_college_count": {"max": 6, "min": 0},
                "cd_dep_count": {"max": 6, "min": 0},
                "cd_dep_employed_count": {"max": 6, "min": 0},
                "cd_purchase_estimate": {"max": 10000, "min": 500},
            },
            "foreign_keys": [],
        },
        "date_dim": {
            "alias": "d",
            "columns": {
                # "d_date": {"max": "2100-01-01", "min": "1900-01-02"},
                "d_date_sk": {"max": 2488070, "min": 2415022},
                "d_dom": {"max": 31, "min": 1},
                "d_dow": {"max": 6, "min": 0},
                "d_first_dom": {"max": 2488070, "min": 2415021},
                "d_fy_quarter_seq": {"max": 801, "min": 1},
                "d_fy_week_seq": {"max": 10436, "min": 1},
                "d_fy_year": {"max": 2100, "min": 1900},
                "d_last_dom": {"max": 2488372, "min": 2415020},
                "d_month_seq": {"max": 2400, "min": 0},
                "d_moy": {"max": 12, "min": 1},
                "d_qoy": {"max": 4, "min": 1},
                "d_quarter_seq": {"max": 801, "min": 1},
                "d_same_day_lq": {"max": 2487978, "min": 2414930},
                "d_same_day_ly": {"max": 2487705, "min": 2414657},
                "d_week_seq": {"max": 10436, "min": 1},
                "d_year": {"max": 2100, "min": 1900},
            },
            "foreign_keys": [],
        },
        "household_demographics": {
            "alias": "hd",
            "columns": {
                "hd_demo_sk": {"max": 7200, "min": 1},
                "hd_dep_count": {"max": 9, "min": 0},
                "hd_income_band_sk": {"max": 20, "min": 1},
                "hd_vehicle_count": {"max": 4, "min": -1},
            },
            "foreign_keys": [],
        },
        "income_band": {
            "alias": "ib",
            "columns": {
                "ib_income_band_sk": {"max": 20, "min": 1},
                "ib_lower_bound": {"max": 190001, "min": 0},
                "ib_upper_bound": {"max": 200000, "min": 10000},
            },
            "foreign_keys": [],
        },
        "inventory": {
            "alias": "inv",
            "columns": {
                "inv_date_sk": {"max": 2452635, "min": 2450815},
                "inv_item_sk": {"max": 204000, "min": 1},
                "inv_quantity_on_hand": {"max": 1000, "min": 0},
                "inv_warehouse_sk": {"max": 15, "min": 1},
            },
            "foreign_keys": [
                {
                    "column": "inv_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "inv_item_sk",
                    "ref_table": "item",
                    "ref_column": "i_item_sk",
                },
                {
                    "column": "inv_warehouse_sk",
                    "ref_table": "warehouse",
                    "ref_column": "w_warehouse_sk",
                },
            ],
        },
        "item": {
            "alias": "i",
            "columns": {
                "i_brand_id": {"max": 10016017, "min": 1001001},
                "i_category_id": {"max": 10, "min": 1},
                "i_class_id": {"max": 16, "min": 1},
                "i_current_price": {"max": 99.97, "min": 0.09},
                "i_item_sk": {"max": 204000, "min": 1},
                "i_manager_id": {"max": 100, "min": 1},
                "i_manufact_id": {"max": 1000, "min": 1},
                # "i_rec_end_date": {"max": "2001-10-26", "min": "1999-10-27"},
                # "i_rec_start_date": {"max": "2001-10-27", "min": "1997-10-27"},
                "i_wholesale_cost": {"max": 89.26, "min": 0.02},
            },
            "foreign_keys": [],
        },
        "promotion": {
            "alias": "p",
            "columns": {
                "p_cost": {"max": 1000.0, "min": 1000.0},
                "p_end_date_sk": {"max": 2450962, "min": 2450112},
                "p_item_sk": {"max": 203506, "min": 98},
                "p_promo_sk": {"max": 1000, "min": 1},
                "p_response_target": {"max": 1, "min": 1},
                "p_start_date_sk": {"max": 2450915, "min": 2450097},
            },
            "foreign_keys": [],
        },
        "reason": {
            "alias": "r",
            "columns": {"r_reason_sk": {"max": 55, "min": 1}},
            "foreign_keys": [],
        },
        "ship_mode": {
            "alias": "sm",
            "columns": {"sm_ship_mode_sk": {"max": 20, "min": 1}},
            "foreign_keys": [],
        },
        "store": {
            "alias": "s",
            "columns": {
                "s_closed_date_sk": {"max": 2451312, "min": 2450828},
                "s_company_id": {"max": 1, "min": 1},
                "s_division_id": {"max": 1, "min": 1},
                "s_floor_space": {"max": 9917607, "min": 5010719},
                "s_gmt_offset": {"max": -5.0, "min": -6.0},
                "s_market_id": {"max": 10, "min": 1},
                "s_number_employees": {"max": 300, "min": 200},
                # "s_rec_end_date": {"max": "2001-03-12", "min": "1999-03-13"},
                # "s_rec_start_date": {"max": "2001-03-13", "min": "1997-03-13"},
                "s_store_sk": {"max": 402, "min": 1},
                "s_tax_precentage": {"max": 0.11, "min": 0.0},
            },
            "foreign_keys": [],
        },
        "store_returns": {
            "alias": "sr",
            "columns": {
                "sr_addr_sk": {"max": 1000000, "min": 1},
                "sr_cdemo_sk": {"max": 1920800, "min": 1},
                "sr_customer_sk": {"max": 2000000, "min": 1},
                "sr_fee": {"max": 100.0, "min": 0.5},
                "sr_hdemo_sk": {"max": 7200, "min": 1},
                "sr_item_sk": {"max": 204000, "min": 1},
                "sr_net_loss": {"max": 10243.17, "min": 0.5},
                "sr_reason_sk": {"max": 55, "min": 1},
                "sr_refunded_cash": {"max": 17186.89, "min": 0.0},
                "sr_return_amt": {"max": 18849.6, "min": 0.0},
                "sr_return_amt_inc_tax": {"max": 19743.64, "min": 0.0},
                "sr_return_quantity": {"max": 100, "min": 1},
                "sr_return_ship_cost": {"max": 9688.28, "min": 0.0},
                "sr_return_tax": {"max": 1614.78, "min": 0.0},
                "sr_return_time_sk": {"max": 61199, "min": 28799},
                "sr_returned_date_sk": {"max": 2452822, "min": 2450820},
                "sr_reversed_charge": {"max": 15723.35, "min": 0.0},
                "sr_store_credit": {"max": 15604.15, "min": 0.0},
                "sr_store_sk": {"max": 400, "min": 1},
                "sr_ticket_number": {"max": 24000000, "min": 3},
            },
            "foreign_keys": [
                {
                    "column": "sr_returned_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "sr_return_time_sk",
                    "ref_table": "time_dim",
                    "ref_column": "t_time_sk",
                },
                {
                    "column": "sr_item_sk",
                    "ref_table": "item",
                    "ref_column": "i_item_sk",
                },
                {
                    "column": "sr_item_sk",
                    "ref_table": "store_sales",
                    "ref_column": "ss_item_sk",
                },
                {
                    "column": "sr_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "sr_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "sr_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "sr_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "sr_store_sk",
                    "ref_table": "store",
                    "ref_column": "s_store_sk",
                },
                {
                    "column": "sr_reason_sk",
                    "ref_table": "reason",
                    "ref_column": "r_reason_sk",
                },
                {
                    "column": "sr_ticket_number",
                    "ref_table": "store_sales",
                    "ref_column": "ss_ticket_number",
                },
            ],
        },
        "store_sales": {
            "alias": "ss",
            "columns": {
                "ss_addr_sk": {"max": 1000000, "min": 1},
                "ss_cdemo_sk": {"max": 1920800, "min": 1},
                "ss_coupon_amt": {"max": 19120.7, "min": 0.0},
                "ss_customer_sk": {"max": 2000000, "min": 1},
                "ss_ext_discount_amt": {"max": 19120.7, "min": 0.0},
                "ss_ext_list_price": {"max": 20000.0, "min": 1.0},
                "ss_ext_sales_price": {"max": 19972.0, "min": 0.0},
                "ss_ext_tax": {"max": 1797.48, "min": 0.0},
                "ss_ext_wholesale_cost": {"max": 10000.0, "min": 1.0},
                "ss_hdemo_sk": {"max": 7200, "min": 1},
                "ss_item_sk": {"max": 204000, "min": 1},
                "ss_list_price": {"max": 200.0, "min": 1.0},
                "ss_net_paid": {"max": 19972.0, "min": 0.0},
                "ss_net_paid_inc_tax": {"max": 21769.48, "min": 0.0},
                "ss_net_profit": {"max": 9986.0, "min": -10000.0},
                "ss_promo_sk": {"max": 1000, "min": 1},
                "ss_quantity": {"max": 100, "min": 1},
                "ss_sales_price": {"max": 200.0, "min": 0.0},
                "ss_sold_date_sk": {"max": 2452642, "min": 2450816},
                "ss_sold_time_sk": {"max": 75599, "min": 28800},
                "ss_store_sk": {"max": 400, "min": 1},
                "ss_ticket_number": {"max": 24000000, "min": 1},
                "ss_wholesale_cost": {"max": 100.0, "min": 1.0},
            },
            "foreign_keys": [
                {
                    "column": "ss_sold_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "ss_sold_time_sk",
                    "ref_table": "time_dim",
                    "ref_column": "t_time_sk",
                },
                {
                    "column": "ss_item_sk",
                    "ref_table": "item",
                    "ref_column": "i_item_sk",
                },
                {
                    "column": "ss_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "ss_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "ss_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "ss_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "ss_store_sk",
                    "ref_table": "store",
                    "ref_column": "s_store_sk",
                },
                {
                    "column": "ss_promo_sk",
                    "ref_table": "promotion",
                    "ref_column": "p_promo_sk",
                },
            ],
        },
        "time_dim": {
            "alias": "t",
            "columns": {
                "t_hour": {"max": 23, "min": 0},
                "t_minute": {"max": 59, "min": 0},
                "t_second": {"max": 59, "min": 0},
                "t_time": {"max": 86399, "min": 0},
                "t_time_sk": {"max": 86399, "min": 0},
            },
            "foreign_keys": [],
        },
        "warehouse": {
            "alias": "w",
            "columns": {
                "w_gmt_offset": {"max": -5.0, "min": -6.0},
                "w_warehouse_sk": {"max": 15, "min": 1},
                "w_warehouse_sq_ft": {"max": 933435, "min": 198821},
            },
            "foreign_keys": [],
        },
        "web_page": {
            "alias": "wp",
            "columns": {
                "wp_access_date_sk": {"max": 2452648, "min": 2452548},
                "wp_char_count": {"max": 8339, "min": 354},
                "wp_creation_date_sk": {"max": 2450815, "min": 2450675},
                "wp_customer_sk": {"max": 1998639, "min": 2535},
                "wp_image_count": {"max": 7, "min": 1},
                "wp_link_count": {"max": 25, "min": 2},
                "wp_max_ad_count": {"max": 4, "min": 0},
                # "wp_rec_end_date": {"max": "2001-09-02", "min": "1999-09-03"},
                # "wp_rec_start_date": {"max": "2001-09-03", "min": "1997-09-03"},
                "wp_web_page_sk": {"max": 2040, "min": 1},
            },
            "foreign_keys": [],
        },
        "web_returns": {
            "alias": "wr",
            "columns": {
                "wr_account_credit": {"max": 21456.03, "min": 0.0},
                "wr_fee": {"max": 100.0, "min": 0.5},
                "wr_item_sk": {"max": 204000, "min": 1},
                "wr_net_loss": {"max": 15455.31, "min": 0.5},
                "wr_order_number": {"max": 6000000, "min": 1},
                "wr_reason_sk": {"max": 55, "min": 1},
                "wr_refunded_addr_sk": {"max": 1000000, "min": 1},
                "wr_refunded_cash": {"max": 25440.96, "min": 0.0},
                "wr_refunded_cdemo_sk": {"max": 1920800, "min": 1},
                "wr_refunded_customer_sk": {"max": 2000000, "min": 1},
                "wr_refunded_hdemo_sk": {"max": 7200, "min": 1},
                "wr_return_amt": {"max": 28310.42, "min": 0.0},
                "wr_return_amt_inc_tax": {"max": 30009.04, "min": 0.0},
                "wr_return_quantity": {"max": 100, "min": 1},
                "wr_return_ship_cost": {"max": 13912.47, "min": 0.0},
                "wr_return_tax": {"max": 2395.95, "min": 0.0},
                "wr_returned_date_sk": {"max": 2453000, "min": 2450821},
                "wr_returned_time_sk": {"max": 86399, "min": 0},
                "wr_returning_addr_sk": {"max": 1000000, "min": 1},
                "wr_returning_cdemo_sk": {"max": 1920800, "min": 1},
                "wr_returning_customer_sk": {"max": 2000000, "min": 1},
                "wr_returning_hdemo_sk": {"max": 7200, "min": 1},
                "wr_reversed_charge": {"max": 23012.08, "min": 0.0},
                "wr_web_page_sk": {"max": 2040, "min": 1},
            },
            "foreign_keys": [
                {
                    "column": "wr_returned_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "wr_returned_time_sk",
                    "ref_table": "time_dim",
                    "ref_column": "t_time_sk",
                },
                {
                    "column": "wr_item_sk",
                    "ref_table": "item",
                    "ref_column": "i_item_sk",
                },
                {
                    "column": "wr_item_sk",
                    "ref_table": "web_sales",
                    "ref_column": "ws_item_sk",
                },
                {
                    "column": "wr_refunded_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "wr_refunded_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "wr_refunded_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "wr_refunded_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "wr_returning_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "wr_returning_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "wr_returning_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "wr_returning_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "wr_web_page_sk",
                    "ref_table": "web_page",
                    "ref_column": "wp_web_page_sk",
                },
                {
                    "column": "wr_reason_sk",
                    "ref_table": "reason",
                    "ref_column": "r_reason_sk",
                },
                {
                    "column": "wr_order_number",
                    "ref_table": "web_sales",
                    "ref_column": "ws_order_number",
                },
            ],
        },
        "web_sales": {
            "alias": "ws",
            "columns": {
                "ws_bill_addr_sk": {"max": 1000000, "min": 1},
                "ws_bill_cdemo_sk": {"max": 1920800, "min": 1},
                "ws_bill_customer_sk": {"max": 1999999, "min": 1},
                "ws_bill_hdemo_sk": {"max": 7200, "min": 1},
                "ws_coupon_amt": {"max": 27913.05, "min": 0.0},
                "ws_ext_discount_amt": {"max": 29767.0, "min": 0.0},
                "ws_ext_list_price": {"max": 30000.0, "min": 1.01},
                "ws_ext_sales_price": {"max": 29943.0, "min": 0.0},
                "ws_ext_ship_cost": {"max": 14915.0, "min": 0.0},
                "ws_ext_tax": {"max": 2673.27, "min": 0.0},
                "ws_ext_wholesale_cost": {"max": 10000.0, "min": 1.0},
                "ws_item_sk": {"max": 204000, "min": 1},
                "ws_list_price": {"max": 300.0, "min": 1.0},
                "ws_net_paid": {"max": 29943.0, "min": 0.0},
                "ws_net_paid_inc_ship": {"max": 42919.59, "min": 0.0},
                "ws_net_paid_inc_ship_tax": {"max": 44853.29, "min": 0.0},
                "ws_net_paid_inc_tax": {"max": 32376.27, "min": 0.0},
                "ws_net_profit": {"max": 19962.0, "min": -10000.0},
                "ws_order_number": {"max": 6000000, "min": 1},
                "ws_promo_sk": {"max": 1000, "min": 1},
                "ws_quantity": {"max": 100, "min": 1},
                "ws_sales_price": {"max": 300.0, "min": 0.0},
                "ws_ship_addr_sk": {"max": 1000000, "min": 1},
                "ws_ship_cdemo_sk": {"max": 1920800, "min": 1},
                "ws_ship_customer_sk": {"max": 2000000, "min": 1},
                "ws_ship_date_sk": {"max": 2452762, "min": 2450817},
                "ws_ship_hdemo_sk": {"max": 7200, "min": 1},
                "ws_ship_mode_sk": {"max": 20, "min": 1},
                "ws_sold_date_sk": {"max": 2452642, "min": 2450816},
                "ws_sold_time_sk": {"max": 86399, "min": 0},
                "ws_warehouse_sk": {"max": 15, "min": 1},
                "ws_web_page_sk": {"max": 2040, "min": 1},
                "ws_web_site_sk": {"max": 24, "min": 1},
                "ws_wholesale_cost": {"max": 100.0, "min": 1.0},
            },
            "foreign_keys": [
                {
                    "column": "ws_sold_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "ws_sold_time_sk",
                    "ref_table": "time_dim",
                    "ref_column": "t_time_sk",
                },
                {
                    "column": "ws_ship_date_sk",
                    "ref_table": "date_dim",
                    "ref_column": "d_date_sk",
                },
                {
                    "column": "ws_item_sk",
                    "ref_table": "item",
                    "ref_column": "i_item_sk",
                },
                {
                    "column": "ws_bill_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "ws_bill_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "ws_bill_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "ws_bill_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "ws_ship_customer_sk",
                    "ref_table": "customer",
                    "ref_column": "c_customer_sk",
                },
                {
                    "column": "ws_ship_cdemo_sk",
                    "ref_table": "customer_demographics",
                    "ref_column": "cd_demo_sk",
                },
                {
                    "column": "ws_ship_hdemo_sk",
                    "ref_table": "household_demographics",
                    "ref_column": "hd_demo_sk",
                },
                {
                    "column": "ws_ship_addr_sk",
                    "ref_table": "customer_address",
                    "ref_column": "ca_address_sk",
                },
                {
                    "column": "ws_web_page_sk",
                    "ref_table": "web_page",
                    "ref_column": "wp_web_page_sk",
                },
                {
                    "column": "ws_web_site_sk",
                    "ref_table": "web_site",
                    "ref_column": "web_site_sk",
                },
                {
                    "column": "ws_ship_mode_sk",
                    "ref_table": "ship_mode",
                    "ref_column": "sm_ship_mode_sk",
                },
                {
                    "column": "ws_warehouse_sk",
                    "ref_table": "warehouse",
                    "ref_column": "w_warehouse_sk",
                },
                {
                    "column": "ws_promo_sk",
                    "ref_table": "promotion",
                    "ref_column": "p_promo_sk",
                },
            ],
        },
        "web_site": {
            "alias": "web",
            "columns": {
                "web_close_date_sk": {"max": 2447131, "min": 2443328},
                "web_company_id": {"max": 6, "min": 1},
                "web_gmt_offset": {"max": -5.0, "min": -6.0},
                "web_mkt_id": {"max": 6, "min": 1},
                "web_open_date_sk": {"max": 2450807, "min": 2450628},
                # "web_rec_end_date": {"max": "2001-08-15", "min": "1999-08-16"},
                # "web_rec_start_date": {"max": "2001-08-16", "min": "1997-08-16"},
                "web_site_sk": {"max": 24, "min": 1},
                "web_tax_percentage": {"max": 0.11, "min": 0.02},
            },
            "foreign_keys": [],
        },
    }

    # validate foreign keys
    for table_name, table_info in tables.items():
        for fk in table_info["foreign_keys"]:
            ref_table = fk["ref_table"]
            if fk["column"] not in tables[table_name]["columns"]:
                raise ValueError(
                    f"Table {table_name} has foreign key column "
                    f"{fk['column']} that does not exist"
                )
            if ref_table not in tables:
                raise ValueError(
                    f"Table {table_name} has foreign key "
                    f"to non-existing table {ref_table}"
                )
            if fk["ref_column"] not in tables[ref_table]["columns"]:
                raise ValueError(
                    f"Table {table_name} has foreign key to column {fk['ref_column']} "
                    f"that does not exist in table {ref_table}"
                )

    fact_tables = [
        "store_sales",
        "store_returns",
        "catalog_sales",
        "catalog_returns",
        "web_sales",
        "web_returns",
        "inventory",
    ]

    return tables, fact_tables


def generate_star_join_with_fact(
    tables: Dict[str, Dict], fact: str, num_signs: int, num_per_sign: int, seed: int
) -> Dict[str, Dict[int, str]]:
    """
    Generate star join queries with a specific fact table. The choices of star-join
    signatures is 2^n, where n is the number of foreign keys in the fact table.
    If 2^n is bounded by `num_signs`, we prioritize to generate queries with joins.
    num_signs = (n choose 0) + (n choose 1) + (n choose 2) + ... + (n choose k) + R,
    where R < (n choose k+1). We will random sample R signatures from (n choose k+1)
    and add them to the query pool.
    """
    fact_table = tables[fact]
    fact_alias = fact_table["alias"]
    from_clause_base = f"{fact} {fact_alias}"

    fks = fact_table["foreign_keys"]
    table_cnt = Counter([fk["ref_table"] for fk in fks])
    alias_list = []
    from_clauses_precomputed = []
    join_clauses_precomputed = []
    alias_key = {}
    for fk in fks:
        tab = fk["ref_table"]
        if table_cnt[tab] == 1:
            alias = tables[tab]["alias"]
            del table_cnt[tab]
        else:
            alias = tables[tab]["alias"] + str(table_cnt[tab] - 1)
            table_cnt[tab] -= 1
        alias_list.append(alias)
        from_clauses_precomputed.append(f"{tab} {alias}")
        join_clauses_precomputed.append(
            f"{fact_alias}.{fk['column']} = {alias}.{fk['ref_column']}"
        )
        alias_key[alias] = (tab, fk["ref_column"])

    gen_queries: Dict[str, Dict[int, str]] = {}
    num_dims = len(fks)
    ttl = 0
    for num_dim in range(1, num_dims + 1):
        if ttl >= num_signs:
            break
        for combo in itertools.combinations(
            zip(alias_list, from_clauses_precomputed, join_clauses_precomputed), num_dim
        ):
            sub_alias = [c[0] for c in combo]
            sub_from = [c[1] for c in combo]
            sub_join = [c[2] for c in combo]
            from_clause = ", ".join([from_clause_base] + sub_from)
            join_clause = " AND ".join(sub_join)
            query_prefix = f"SELECT COUNT(*) FROM {from_clause} WHERE {join_clause}"

            gen_queries[f"{fact_alias}{ttl}"] = {}
            np.random.seed(seed + ttl)
            for i in range(num_per_sign):
                # do not precompute in the loop to avoid randomness
                filter_clause = []
                num_pred = np.random.randint(low=1, high=num_dim + 1)
                pred_aliases = np.random.choice(sub_alias, size=num_pred, replace=False)
                for pred_alias in pred_aliases:
                    # Randomly pick a numeric column (excluding the foreign key column).
                    pred_tab, pred_tab_key = alias_key[pred_alias]
                    col_info = tables[pred_tab]["columns"]
                    pred_col = np.random.choice(list(col_info.keys()))

                    # Randomly pick an operator among <, <=, >, >=.
                    # Note:
                    # 1. I dropped "=" to reduce the chance to get an empty result
                    # 2. Extensions can be made to support more operators

                    op = np.random.choice(["<", "<=", ">", ">="])

                    # randomly pick an operator among
                    if isinstance(col_info[pred_col]["min"], int):
                        val = np.random.randint(
                            low=col_info[pred_col]["min"],
                            high=col_info[pred_col]["max"] + 1,
                        )
                    elif isinstance(col_info[pred_col]["min"], float):
                        val = np.random.uniform(
                            low=col_info[pred_col]["min"],
                            high=col_info[pred_col]["max"],
                        )
                    else:
                        raise ValueError(
                            f"Unsupported data type {type(col_info[pred_col]['min'])}"
                        )
                    filter_clause.append(f"{pred_alias}.{pred_col} {op} {val}")

                query = (
                    query_prefix
                    if num_pred == 0
                    else f"{query_prefix} AND {' AND '.join(filter_clause)}"
                )
                gen_queries[f"{fact_alias}{ttl}"][i] = query
                # print(query)
            ttl += 1
            if ttl >= num_signs:
                break
    return gen_queries


def generate_star_join(
    tables: Dict[str, Dict],
    fact_tables: List[str],
    num_signs: int,
    num_per_sign: int,
    seed: int,
) -> Dict[str, Dict[int, str]]:
    """
    Generate star join queries with.
    1. a fact table with n foreign keys can generate 2^n star-join signatures.
    2. per sign, we generate num_per_sign queries by adjusting the number
       and content of column predicates

    TPC-DS has 7 fact tables, each with 9, 11, 17, 18, 17, 15, 3 foreign keys.
    The total number of star-join signature is
    559624.

    The number of signatures of each fact table is bounded by `num_signs`.
    The number of total queries generated is bounded by
    `num_signs * num_per_sign * n_fact_tables`.

    """
    gen_queries: Dict[str, Dict[int, str]] = {}
    for fact in fact_tables:
        gen_queries_fact = generate_star_join_with_fact(
            tables, fact, num_signs, num_per_sign, seed
        )
        gen_queries.update(gen_queries_fact)
    return gen_queries


if __name__ == "__main__":
    tables, fact_tables = get_table_info()

    gen_queries = generate_star_join(tables, fact_tables, 1000, 10, 0)

    sql_path = "spark-sqls"
    JsonHandler.dump_to_file(gen_queries, f"{sql_path}/collection.json", indent=2)

    num_queries = sum([len(v) for v in gen_queries.values()])
    print(f"Generated {len(gen_queries)} queries")

    for sign, queries in gen_queries.items():
        os.makedirs(f"{sql_path}/{sign}", exist_ok=True)
        for vid, query in queries.items():
            with open(f"{sql_path}/{sign}/{sign}-{vid}.sql", "w") as f:
                f.write(query)
