"""The mushroom body's memory circuit in FlyWire v783 (both hemispheres), by cell class: who writes, who reads, who gates.
Classes from consolidated_cell_types.csv.gz: KC (Kenyon cells), MBON, DAN (PAM + PPL1), APL, DPM, PN (uniglomerular and
multiglomerular projection neurons), OTHER (everything else)."""
import pandas as pd

D = "/games/ailab/lab/flybrain-lab/data/flywire783/"
t = pd.read_csv(D + "consolidated_cell_types.csv.gz")
pt = t["primary_type"].fillna("")
cls = pd.Series("OTHER", index=t["root_id"].values)
for name, pat in [("KC", r"^KC"), ("MBON", r"^MBON"), ("DAN", r"^(PAM|PPL1)"), ("APL", r"^APL$"), ("DPM", r"^DPM$"), ("PN", r"(^M_|_(ad|l|il|v|vl|a)?PN$)")]:
    cls[t["root_id"].values[pt.str.match(pat).values]] = name
typ = pd.Series(pt.values, index=t["root_id"].values)
c = pd.read_csv(D + "connections.csv.gz")
c["pre_cls"] = c["pre_root_id"].map(cls).fillna("OTHER"); c["post_cls"] = c["post_root_id"].map(cls).fillna("OTHER")
n_cells = cls.value_counts()
print("cells:", n_cells.to_dict())
order = ["PN", "KC", "APL", "DPM", "DAN", "MBON", "OTHER"]
m = c.pivot_table(index="pre_cls", columns="post_cls", values="syn_count", aggfunc="sum", fill_value=0).reindex(index=order, columns=order, fill_value=0)
print("\nsynapses pre (rows) -> post (columns):")
print(m.to_string())
mb = c[(c.pre_cls != "OTHER") | (c.post_cls != "OTHER")]
# inputs to each class, as shares
for cl in ("KC", "MBON", "DAN", "APL"):
    inp = c[c.post_cls == cl].groupby("pre_cls")["syn_count"].sum().sort_values(ascending=False)
    out = c[c.pre_cls == cl].groupby("post_cls")["syn_count"].sum().sort_values(ascending=False)
    print(f"\n{cl}: inputs {int(inp.sum()):,} syn: " + ", ".join(f"{k} {v / inp.sum():.1%}" for k, v in inp.items())
          + f"\n{cl}: outputs {int(out.sum()):,} syn: " + ", ".join(f"{k} {v / out.sum():.1%}" for k, v in out.items()))
# KC claws: distinct PN partners per KC, and synapses per PN-KC pair
pk = c[(c.pre_cls == "PN") & (c.post_cls == "KC")].groupby(["post_root_id"]).agg(n_pn=("pre_root_id", "nunique"), syn=("syn_count", "sum"))
print(f"\nKC: distinct PN partners per KC mean {pk.n_pn.mean():.1f} (median {pk.n_pn.median():.0f}), PN synapses per KC mean {pk.syn.mean():.0f}; "
      f"KCs with any PN input {len(pk)} of {n_cells['KC']}")
pnk = c[(c.pre_cls == "PN") & (c.post_cls == "KC")].groupby("pre_root_id")["post_root_id"].nunique()
print(f"PN: distinct KC targets per PN mean {pnk.mean():.0f}, median {pnk.median():.0f}")
# per MBON type: input shares and feedback to DANs
mi = c[c.post_cls == "MBON"].assign(ty=lambda d: d.post_root_id.map(typ)).groupby(["ty", "pre_cls"])["syn_count"].sum().unstack(fill_value=0)
mi = (mi.T / mi.sum(1)).T
mo = c[(c.pre_cls == "MBON")].assign(ty=lambda d: d.pre_root_id.map(typ)).groupby(["ty", "post_cls"])["syn_count"].sum().unstack(fill_value=0)
mo = (mo.T / mo.sum(1)).T
print("\nper MBON type: share of input from KC / DAN / other MBON, and share of output to DAN / MBON:")
for ty in sorted(mi.index):
    print(f"  {ty:7s} in: KC {mi.loc[ty].get('KC', 0):.0%} DAN {mi.loc[ty].get('DAN', 0):.0%} MBON {mi.loc[ty].get('MBON', 0):.0%} APL {mi.loc[ty].get('APL', 0):.0%} | out: DAN {mo.loc[ty].get('DAN', 0):.0%} MBON {mo.loc[ty].get('MBON', 0):.0%} KC {mo.loc[ty].get('KC', 0):.0%}")
di = c[c.post_cls == "DAN"].assign(ty=lambda d: d.post_root_id.map(typ)).groupby(["ty", "pre_cls"])["syn_count"].sum().unstack(fill_value=0)
di = (di.T / di.sum(1)).T
do = c[c.pre_cls == "DAN"].assign(ty=lambda d: d.pre_root_id.map(typ)).groupby(["ty", "post_cls"])["syn_count"].sum().unstack(fill_value=0)
do = (do.T / do.sum(1)).T
print("\nper DAN type: share of input from MBON / DAN / KC, and share of output to KC / MBON / DAN:")
for ty in sorted(di.index):
    print(f"  {ty:7s} in: MBON {di.loc[ty].get('MBON', 0):.0%} DAN {di.loc[ty].get('DAN', 0):.0%} KC {di.loc[ty].get('KC', 0):.0%} | out: KC {do.loc[ty].get('KC', 0):.0%} MBON {do.loc[ty].get('MBON', 0):.0%} DAN {do.loc[ty].get('DAN', 0):.0%} APL {do.loc[ty].get('APL', 0):.0%}")
# KC -> MBON convergence: distinct KCs per MBON cell, and KC -> KC
km = c[(c.pre_cls == "KC") & (c.post_cls == "MBON")].groupby("post_root_id")["pre_root_id"].nunique()
print(f"\nKC->MBON: distinct KCs per MBON cell mean {km.mean():.0f}, median {km.median():.0f} (of about {n_cells['KC'] // 2} per side)")
kk = c[(c.pre_cls == "KC") & (c.post_cls == "KC")]
print(f"KC->KC: {int(kk.syn_count.sum()):,} synapses over {len(kk):,} pairs; KC->APL {int(m.loc['KC', 'APL']):,}, APL->KC {int(m.loc['APL', 'KC']):,}; DAN->KC {int(m.loc['DAN', 'KC']):,}; KC->DAN {int(m.loc['KC', 'DAN']):,}")
