theory AnyPos
  imports Complex_Main
begin

datatype Zonotope = Zonotope real "real list"

datatype DiffZonotope = DiffZonotope Zonotope Zonotope Zonotope

fun diff_zono_wellformed :: "DiffZonotope \<Rightarrow> bool"
  where "diff_zono_wellformed (DiffZonotope (Zonotope _ gs1) (Zonotope _ gsd) (Zonotope _ gs2)) =
          ((length gs1 = length gsd) \<and> (length gsd = length gs2))"

text\<open>
We consider 1-dimensional Zonotopes (a.k.a. affine forms).
\<close>

fun zono_concretization :: "Zonotope \<Rightarrow> real set"
  where "zono_concretization (Zonotope c gs) = {
          c + sum_list (map2 (*) s gs) |
          s. (length s = length gs) \<and>
          (\<forall> x \<in> set s . -1 \<le> x \<and> x \<le> 1 )}"

fun diff_zono_concretization :: "DiffZonotope \<Rightarrow> (real\<times>real\<times>real) set"
  where "diff_zono_concretization (DiffZonotope (Zonotope c1 g1) (Zonotope cd gd) (Zonotope c2 g2)) =
          {
            (
              (c1 + sum_list (map2 (*) s g1)),
              (cd + sum_list (map2 (*) s gd)),
              (c2 + sum_list (map2 (*) s g2))
            ) |
              s . (length s = length g1) \<and>
                (\<forall> x \<in> set s . -1 \<le> x \<and> x \<le> 1 )
          }"


text\<open>
Simple sanity check
\<close>

lemma lets_get_started:
  assumes "l \<le> x \<and> x \<le> u"
      and "l < u"
  shows "x \<in> (zono_concretization (Zonotope ((u+l)/2) [((u-l)/2)]))"
  unfolding zono_concretization.simps
  apply simp
  apply (intro exI[where x="[(x-(u+l)/2)/((u-l)/2)]"])
  using assms
  apply auto
    apply argo
   apply (simp add: field_simps)
  by (simp add: field_simps)


lemma zono_not_empty:
  "(zono_concretization z) \<noteq> empty"
  proof (induct z)
    case (Zonotope x1a x2a)
    then have "\<exists>s :: real list. length s = length x2a \<and> (\<forall>x\<in>set s. - 1 \<le> x \<and> x \<le> 1)"
      apply (intro exI[of _ "replicate (length x2a) (0::real)"])
      by simp
    then show ?case
      unfolding zono_concretization.simps
      by auto
  qed

text\<open>
Lower and Upper Bounds
\<close>

fun zono_lower :: "Zonotope \<Rightarrow> real"
  where "zono_lower (Zonotope c gs) = c + sum_list (map (uminus \<circ> abs) gs)"

fun zono_upper :: "Zonotope \<Rightarrow> real"
  where "zono_upper (Zonotope c gs) = c + sum_list (map abs gs)"

lemma pointwise_abs_lower:
  assumes "-1 \<le> s"
      and "s \<le> 1"
    shows "-(abs (a::real)) \<le> s * a"
proof (cases "a \<ge> 0")
  case True  
  then show ?thesis
    by (metis abs_ge_minus_self abs_if assms(1) divide_self_if less_eq_real_def mult_cancel_right1 order_less_asym
        pos_minus_divide_le_eq)
next
  case False
  then show ?thesis
    by (simp add: assms(2))
qed

lemma pointwise_abs_upper:
  assumes "-1 \<le> s"
      and "s \<le> 1"
    shows "s * a \<le> (abs (a::real))"
proof (cases "a \<ge> 0")
  case True  
  then show ?thesis
    by (simp add: assms(2) mult_le_cancel_right2)
next
  case False
  then show ?thesis
    by (metis abs_ge_minus_self assms(1) linorder_linear mult_minus1 mult_right_mono_neg order_trans)
qed


lemma sum_list_abs_lower:
    fixes xs :: "real list"
    fixes s :: "real list"
  assumes "length s = length xs"
      and "\<forall>x\<in>set s. -1 \<le> x \<and> x \<le> 1"
    shows "sum_list (map (uminus \<circ> abs) xs) \<le> sum_list (map2 (*) s xs)"
  using assms
proof (induct s xs rule: list_induct2)
  case Nil then show ?case by simp
next
  case (Cons a s b xs)
  have "\<forall>x\<in>set s. - 1 \<le> x \<and> x \<le> 1"
    using Cons by auto
  then have "sum_list (map (uminus \<circ> abs) xs) \<le> sum_list (map2 (*) s xs)"
    using Cons
    by blast
  moreover have "-(abs b) \<le> a * b" using Cons.prems pointwise_abs_lower by auto
  ultimately show ?case
    by (simp add: comp_def)
qed


lemma sum_list_abs_upper:
    fixes xs :: "real list"
    fixes s :: "real list"
  assumes "length s = length xs"
      and "\<forall>x\<in>set s. -1 \<le> x \<and> x \<le> 1"
    shows "sum_list (map2 (*) s xs) \<le> sum_list (map abs xs)"
  using assms
proof (induct s xs rule: list_induct2)
  case Nil then show ?case by simp
next
  case (Cons a s b xs)
  have "\<forall>x\<in>set s. - 1 \<le> x \<and> x \<le> 1"
    using Cons by auto
  then have "sum_list (map2 (*) s xs) \<le> sum_list (map abs xs)"
    using Cons
    by blast
  moreover have "a * b \<le> (abs b)" using Cons.prems pointwise_abs_upper by auto
  ultimately show ?case
    by (simp add: comp_def)
qed

lemma zono_lower_sound:
  "zono_lower z \<le> Inf (zono_concretization z)"
  proof (induct z)
    case (Zonotope c gs)
    then show ?case
    proof -
      let ?S = "zono_concretization (Zonotope c gs)"
      have lb: "\<And>y. y \<in> ?S \<Longrightarrow> c + sum_list (map (uminus \<circ> abs) gs) \<le> y"
      proof -
        fix y assume "y \<in> ?S"
        then obtain s where s: "y = c + sum_list (map2 (*) s gs)"
            "length s = length gs" "\<forall>x\<in>set s. -1 \<le> x \<and> x \<le> 1"
          by (auto simp: zono_concretization.simps)
        from sum_list_abs_lower[OF s(2) s(3)]
        show "c + sum_list (map (uminus \<circ> abs) gs) \<le> y" by (simp add: s(1))
      qed
      then have "c + sum_list (map (uminus \<circ> abs) gs) \<le> Inf ?S"
        using zono_not_empty[of "(Zonotope c gs)"]
        using cInf_greatest by blast
      then show ?thesis by simp
    qed
  qed

lemma zono_upper_sound:
  "Sup (zono_concretization z) \<le> zono_upper z"
  proof (induct z)
    case (Zonotope c gs)
    then show ?case
    proof -
      let ?S = "zono_concretization (Zonotope c gs)"
      have ub: "\<And>y. y \<in> ?S \<Longrightarrow> y \<le> c + sum_list (map abs gs)"
      proof -
        fix y assume "y \<in> ?S"
        then obtain s where s: "y = c + sum_list (map2 (*) s gs)"
            "length s = length gs" "\<forall>x\<in>set s. -1 \<le> x \<and> x \<le> 1"
          by (auto simp: zono_concretization.simps)
        from sum_list_abs_upper[OF s(2) s(3)]
        show "y \<le> c + sum_list (map abs gs)"
          by (simp add: s(1))
      qed
      then have "Sup ?S \<le> c + sum_list (map abs gs)"
        using zono_not_empty[of "(Zonotope c gs)"]
        using cSup_least by blast
      then show ?thesis by simp
    qed
  qed

  text\<open>
Pos-Any Transformer
\<close>

fun diff_zono_any_pos :: "DiffZonotope \<Rightarrow> DiffZonotope"
  where "diff_zono_any_pos (DiffZonotope (Zonotope c1 g1) (Zonotope cd gd) (Zonotope c2 g2)) = 
      (let
        l1 = (zono_lower (Zonotope c1 g1));
        u1 = (zono_upper (Zonotope c1 g1));
        l2 = (zono_lower (Zonotope c2 g2));
        u2 = (zono_upper (Zonotope c2 g2));
        lam1 = (u1/(u1-l1));
        lamd = (-l1/(u1-l1));
        mu1 = 0.5*lam1*l1;
        mud = 0.5*lamd*u1
      in (DiffZonotope
        (Zonotope (lam1*c1 - mu1) ( mu1 # (map ((*) lam1) g1)))
        (Zonotope (cd - lamd*c1 + mud) ((-mud) # (map2 (\<lambda> x y . (x - lamd*y)) gd g1)))
        (Zonotope c2 ((0::real) # g2))
      ))"

definition relu :: "real \<Rightarrow> real" where
  "relu x = max 0 x"

text\<open>
Soundness of pos-any case
\<close>

(* ============================================================
   Auxiliary lemmas by Claude Opus
   ============================================================ *)

lemma sum_list_map2_scale:
  fixes s gs :: "real list" and c :: real
  assumes "length s = length gs"
  shows "sum_list (map2 (*) s (map ((*) c) gs)) = c * sum_list (map2 (*) s gs)"
  using assms by (induct s gs rule: list_induct2) (auto simp: algebra_simps)

lemma sum_list_map2_sub_scale:
  fixes s gd g1 :: "real list" and c :: real
  assumes "length s = length gd" "length gd = length g1"
  shows "sum_list (map2 (*) s (map2 (\<lambda>a b. a - c * b) gd g1)) =
         sum_list (map2 (*) s gd) - c * sum_list (map2 (*) s g1)"
  using assms
proof (induct s gd arbitrary: g1 rule: list_induct2)
  case Nil then show ?case by simp
next
  case (Cons a s b gd)
  then obtain g g1' where "g1 = g # g1'" "length gd = length g1'"
    by (cases g1) auto
  then show ?case using Cons by (auto simp: algebra_simps)
qed

lemma diff_zono_pos_any:
  fixes z1 :: "Zonotope"
  fixes zd :: "Zonotope"
  fixes z2 :: "Zonotope"
  fixes x :: real
  fixes y :: real
  defines "ZD \<equiv> (DiffZonotope z1 zd z2)"
  assumes ZD_wellformed: "diff_zono_wellformed ZD"
      and zono1_range: "(zono_lower z1) < 0 \<and> 0 < (zono_upper z1)"
      and z2_pos: "0 \<le> (zono_lower z2)"
      and x_y_contained: "(x, (x-y), y) \<in> (diff_zono_concretization ZD)"
    shows "(relu x, ((relu x) - (relu y)), relu y) \<in> (diff_zono_concretization (diff_zono_any_pos ZD))"
proof -
  (* --- Destructure --- *)
  obtain c1 g1 where z1_eq: "z1 = Zonotope c1 g1" by (cases z1)
  obtain cd gd where zd_eq: "zd = Zonotope cd gd" by (cases zd)
  obtain c2 g2 where z2_eq: "z2 = Zonotope c2 g2" by (cases z2)

  from wf have len12: "length g1 = length gd" and len23: "length gd = length g2"
    using ZD_def ZD_wellformed diff_zono_wellformed.simps z1_eq z2_eq zd_eq apply blast
    using ZD_def ZD_wellformed z1_eq z2_eq zd_eq by auto

  (* --- Extract witness s --- *)
  from x_y_contained obtain s where
    x_eq:  "x = c1 + sum_list (map2 (*) s g1)" and
    d_eq:  "x - y = cd + sum_list (map2 (*) s gd)" and
    y_eq:  "y = c2 + sum_list (map2 (*) s g2)" and
    s_len: "length s = length g1" and
    s_bds: "\<forall>v\<in>set s. -1 \<le> v \<and> v \<le> 1"
    unfolding ZD_def z1_eq zd_eq z2_eq diff_zono_concretization.simps
    by auto

  have s_len_gd: "length s = length gd" using s_len len12 by simp

  (* --- Bounds on x --- *)
  define l1 where "l1 = zono_lower (Zonotope c1 g1)"
  define u1 where "u1 = zono_upper (Zonotope c1 g1)"

  from zono1_range have l1_neg: "l1 < 0" and u1_pos: "0 < u1"
    unfolding l1_def u1_def z1_eq by auto
  have diff_pos: "u1 - l1 > 0" using l1_neg u1_pos by auto
  have u1_ne: "u1 \<noteq> 0" and l1_ne: "l1 \<noteq> 0" and diff_ne: "u1 - l1 \<noteq> 0"
    using u1_pos l1_neg diff_pos by auto

  have x_ge_l1: "l1 \<le> x"
    using sum_list_abs_lower[OF s_len s_bds] unfolding x_eq l1_def by simp
  have x_le_u1: "x \<le> u1"
    using sum_list_abs_upper[OF s_len s_bds] unfolding x_eq u1_def by simp

  (* --- y \<ge> 0 --- *)
  have y_nn: "y \<ge> 0"
  proof -
    have "length s = length g2" using s_len len12 len23 by simp
    then have "zono_lower (Zonotope c2 g2) \<le> y"
      using sum_list_abs_lower[of s g2] s_bds y_eq by simp
    then show ?thesis using z2_pos z2_eq by auto
  qed
  then have relu_y: "relu y = y" unfolding relu_def by auto

  (* --- relu x - relu y = relu(-x) + (x - y) --- *)
  have relu_diff: "relu x - relu y = relu (- x) + (x - y)"
  proof -
    have "relu x - relu y = relu x - y" using relu_y by simp
    also have "\<dots> = (relu x - x) + (x - y)" by simp
    also have "relu x - x = relu (- x)"
      unfolding relu_def by auto
    finally show ?thesis by simp
  qed

  (* --- Define transformer parameters --- *)
  define lam1 where "lam1 = u1 / (u1 - l1)"
  define lamd where "lamd = (- l1) / (u1 - l1)"
  define mu1  where "mu1  = (1/2) * lam1 * l1"
  define mud  where "mud  = (1/2) * lamd * u1"

  (* --- Define s0 and new witness --- *)
  define s0 where "s0 = (if x \<ge> 0 then 1 - 2 * x / u1 else 1 - 2 * x / l1)"
  define s' where "s' = s0 # s"

  (* --- s0 \<in> [-1, 1] --- *)
  have s0_ge: "- 1 \<le> s0"
  proof (cases "x \<ge> 0")
    case True
    from x_le_u1 u1_pos have "2 * x / u1 \<le> 2"
      by (auto simp: divide_simps)
    then show ?thesis unfolding s0_def using True by auto
  next
    case False
    from x_ge_l1 l1_neg have "x / l1 \<le> 1"
      by (auto simp: divide_simps)
    then have "2 * x / l1 \<le> 2" by auto
    then show ?thesis unfolding s0_def using False by auto
  qed

  have s0_le: "s0 \<le> 1"
  proof (cases "x \<ge> 0")
    case True
    from True u1_pos have "0 \<le> 2 * x / u1" by auto
    then show ?thesis unfolding s0_def using True by auto
  next
    case False
    then have "x < 0" by auto
    with l1_neg have "0 < x / l1"
      by (auto simp: divide_simps)
    then show ?thesis unfolding s0_def using \<open>x < 0\<close> by auto
  qed

  have s'_bds: "\<forall>v\<in>set s'. -1 \<le> v \<and> v \<le> 1"
    unfolding s'_def using s_bds s0_ge s0_le by auto
  have s'_len: "length s' = Suc (length g1)"
    unfolding s'_def using s_len by auto

  (* === Algebraic identities === *)

  have comp1: "lam1 * x + mu1 * (s0 - 1) = relu x"
  proof (cases "x \<ge> 0")
    case True
    then have "s0 - 1 = - (2 * x / u1)" unfolding s0_def by auto
    moreover have "lam1 * x + mu1 * (- (2 * x / u1)) = x"
      unfolding mu1_def lam1_def using diff_ne u1_ne
      apply simp
      by (metis (mono_tags, lifting) diff_divide_distrib diff_ne divide_divide_eq_right nonzero_mult_div_cancel_left
          right_diff_distrib zero_neq_numeral)
    ultimately show ?thesis using True unfolding relu_def by auto
  next
    case False
    then have "x < 0" by auto
    then have "s0 - 1 = - (2 * x / l1)" unfolding s0_def by auto
    moreover have "lam1 * x + mu1 * (- (2 * x / l1)) = 0"
      unfolding mu1_def lam1_def using diff_ne l1_ne
      by (auto simp: field_simps)
    ultimately show ?thesis using \<open>x < 0\<close> unfolding relu_def by auto
  qed

  have comp2: "- lamd * x + mud * (1 - s0) = relu (- x)"
  proof (cases "x \<ge> 0")
    case True
    then have "1 - s0 = 2 * x / u1" unfolding s0_def by auto
    moreover have "- lamd * x + mud * (2 * x / u1) = 0"
      unfolding mud_def lamd_def using diff_ne u1_ne
      by (auto simp: field_simps)
    ultimately show ?thesis using True unfolding relu_def by auto
  next
    case False
    then have "x < 0" by auto
    then have "1 - s0 = 2 * x / l1" unfolding s0_def by auto
    moreover have "- lamd * x + mud * (2 * x / l1) = - x"
      unfolding mud_def lamd_def using diff_ne l1_ne
      apply simp
      by (metis diff_divide_distrib[of l1 u1 "u1 - l1"] diff_ne divide_eq_minus_1_iff[of "- (u1 - l1)" "u1 - l1"]
          minus_diff_eq[of u1 l1] mult.commute[of x "2"] mult.commute[of x "l1 / (u1 - l1)"]
          mult.commute[of x "u1 / (u1 - l1)"] mult.right_neutral[of "2"] mult.right_neutral[of x]
          mult_minus_right[of x "1"] nonzero_mult_divide_mult_cancel_left[of "2" "1" "u1 - l1"]
          right_diff_distrib[of "2" u1 l1] right_diff_distrib[of x "l1 / (u1 - l1)" "u1 / (u1 - l1)"]
          times_divide_eq_left[of l1 "u1 - l1" x] times_divide_eq_left[of u1 "u1 - l1" "x * 1"]
          times_divide_eq_right[of x "2" "2 * (u1 - l1)"] times_divide_eq_right[of u1 "x * 2" "2 * (u1 - l1)"]
          times_divide_eq_right[of "2" x "2 * (u1 - l1)"] times_divide_eq_right[of u1 "x * 1" "u1 - l1"]
          times_divide_eq_right[of x "1" "u1 - l1"] zero_neq_numeral[of "num.Bit0 num.One"])
    ultimately show ?thesis using \<open>x < 0\<close> unfolding relu_def by auto
  qed

  (* === Component equations === *)

  have tgt1: "relu x = lam1 * c1 - mu1 +
      sum_list (map2 (*) s' (mu1 # map ((*) lam1) g1))"
  proof -
    have "sum_list (map2 (*) s' (mu1 # map ((*) lam1) g1))
          = s0 * mu1 + sum_list (map2 (*) s (map ((*) lam1) g1))"
      unfolding s'_def by simp
    also have "\<dots> = s0 * mu1 + lam1 * sum_list (map2 (*) s g1)"
      using sum_list_map2_scale[OF s_len] by simp
    finally show ?thesis using comp1 x_eq by (auto simp: algebra_simps)
  qed

  have tgt2: "relu (- x) + (x - y) = cd - lamd * c1 + mud +
      sum_list (map2 (*) s' ((- mud) # map2 (\<lambda>a b. a - lamd * b) gd g1))"
  proof -
    have "sum_list (map2 (*) s' ((- mud) # map2 (\<lambda>a b. a - lamd * b) gd g1))
          = s0 * (- mud) + sum_list (map2 (*) s (map2 (\<lambda>a b. a - lamd * b) gd g1))"
      unfolding s'_def by simp
    also have "\<dots> = - s0 * mud +
        (sum_list (map2 (*) s gd) - lamd * sum_list (map2 (*) s g1))"
      using sum_list_map2_sub_scale[OF s_len_gd len12[symmetric]] by simp
    finally show ?thesis 
      unfolding comp2[symmetric]
      unfolding d_eq
      unfolding x_eq
      apply (simp add: sum_list_map2_scale sum_list_map2_sub_scale)
      by argo
  qed

  have tgt3: "relu y = c2 + sum_list (map2 (*) s' (0 # g2))"
  proof -
    have "sum_list (map2 (*) s' (0 # g2)) = sum_list (map2 (*) s g2)"
      unfolding s'_def by simp
    then show ?thesis using y_eq relu_y by auto
  qed

  (* === Show the transformed zonotope matches our construction === *)
  have tf_eq: "diff_zono_any_pos ZD =
    DiffZonotope
      (Zonotope (lam1 * c1 - mu1) (mu1 # map ((*) lam1) g1))
      (Zonotope (cd - lamd * c1 + mud) ((- mud) # map2 (\<lambda>a b. a - lamd * b) gd g1))
      (Zonotope c2 (0 # g2))"
    unfolding ZD_def z1_eq zd_eq z2_eq
      lam1_def lamd_def mu1_def mud_def l1_def u1_def
      diff_zono_any_pos.simps
      Let_def
    by simp

  (* === Final assembly === *)
  have "(relu x, relu (- x) + (x - y), relu y)
        \<in> diff_zono_concretization (diff_zono_any_pos ZD)"
    unfolding tf_eq diff_zono_concretization.simps
    using tgt1 tgt2 tgt3 s'_len s'_bds len12
    by (auto intro: exI[of _ s'])

  then show ?thesis using relu_diff by auto
qed


end