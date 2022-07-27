#include "itensor/all.h"

using namespace itensor;
using std::max;
using std::min;
using std::string;
using std::vector;

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printfln("Usage: %s <input_file>", argv[0]);
    return 0;
  }
  auto in = InputGroup(argv[1], "input");

  // Input parameters
  auto intfile = in.getString("intfile", "Ham1c");
  auto quiet = in.getYesNo("quiet_dmrg", false);
  auto R = in.getInt("separation", 0); // uniform separation of ions
  auto nsweeps = in.getInt("nsweeps");
  auto table = InputGroup(in, "sweeps");
  auto sweeps = Sweeps(nsweeps, table);

  auto ampo_cutoff = in.getReal("ampo_cutoff", 1E-12);

  //
  // Read from Ham1c
  //
  auto input = std::ifstream(intfile);
  if (!input.is_open())
    Error(format("Error opening file %s", intfile));

  int Nx = 0;
  input >> Nx;
  println("Nx=", Nx);

  int N = Nx;

  Real Vpp = 0;
  input >> Vpp;
  println("Vpp=", Vpp);

  Hubbard sites;
  if (fileExists("sites"))
  {
    println("Reading sites from disk");
    sites = readFromFile<Hubbard>("sites");
  }
  else
  {
    sites = Hubbard(N);
    println("Writing sites to disk");
    writeToFile("sites", sites);
  }

  // Make single-particle MPO
  // One- and two-site terms:
  auto ampoH2 = AutoMPO(sites);

  int i1 = 0, i2 = 0;
  Real tV = 0;

  auto count2 = 0;
  while (!input.eof())
  {
    input >> i1 >> i2 >> tV;
    // std::cout<<i1<<" "<<i2<<" "<<tV<<std::endl;
    // If we hit the V's, break and go on to making those
    if (i1 == 1 && i2 == 1 && count2 > 1)
    {
      break;
    }
    if (i1 == i2)
    {
      // fermion electrons
      ampoH2 += tV, "Ntot", i1;
    }
    else
    {
      // fermion electrons
      ampoH2 += tV, "Cdagup", i1, "Cup", i2;
      ampoH2 += tV, "Cdagdn", i1, "Cdn", i2;
    }
    ++count2;
  }
  input.close();
  printfln("Got %d 2-site integrals", count2);
  printfln("%d terms in ampoH2", ampoH2.terms().size());

  println("Creating H2 IQMPO");

  auto H2 = toMPO<IQTensor>(ampoH2, {"Exact=", true});

  // Make interaction MPO
  if (!fileExists("Vcompressed"))
    Error("File 'Vcompressed' not found");

  println("Making V from file 'Vcompressed'");
  auto inV = std::ifstream("Vcompressed");

  // First line of Vcompressed gives N, check with N in integral file
  int NV = 0;
  inV >> NV;
  if (NV != N)
    Error("N from 'Vcompressed' does not match N from integral file");

  auto Zero = QN("Nf=", 0, "Sz=", 0);

  auto links = vector<IQIndex>(1 + N);
  links.at(0) = IQIndex("L0", Index("l0", 1), Zero);

  // construct MPO from the info of each tensor in Vcompressed
  auto V = IQMPO(sites);
  for (auto n : range1(N))
  {
    auto &W = V.Aref(n);
    auto s = sites(n);

    int nn = 0, nr = 0, nc = 0;
    inV >> nn >> nr >> nc;

    links.at(n) = IQIndex(nameint("L", n), Index(nameint("l", n), nc), Zero);

    auto &row = links.at(n - 1);
    auto &col = links.at(n);

    W = IQTensor(dag(row), col, dag(s), prime(s));
    auto T = IQTensor(dag(row), col);

    for (auto nop : range1(3))
    {
      string Op;
      inV >> Op;

      int num = 0;
      inV >> num;

      auto M = Matrix(nr, nc);
      for (auto k : range1(num))
      {
        int r = 0, c = 0;
        inV >> r >> c;
        inV >> M(r - 1, c - 1);
      }
      W += (T + matrixTensor(M, row.index(1), col.index(1))) * sites.op(Op, n);
    }
  }
  inV.close();

  V.Aref(1) *= setElt(links.at(0)(1));
  V.Aref(N) *= setElt(dag(links.at(N)(1)));

  printfln("Maximum bond dimension of H2 IQMPO is %d", maxM(H2));
  printfln("Maximum bond dimension of V IQMPO is %d", maxM(V));

  // Initialize the wavefunction
  IQMPS psi;
  if (fileExists("psi"))
  {
    println("Reading wavefunction from file 'psi'");
    psi = readFromFile<IQMPS>("psi", sites);
  }
  else
  {
    // Make a new initial state
    auto state = InitState(sites, "Emp");

    // different distances from origin if even number of ions in chain
    int R_left = 0;
    int R_right = 0;
    if (R % 2 == 0)
    {
      R_left = R / 2;
      R_right = R / 2;
    }
    else
    {
      R_right = (R + 1) / 2;
      R_left = (R - 1) / 2;
    }

    // if odd number of ions in chain, it is simpler.
    // H3 chain example
    // use +/- 1 to prevent overwriting the site state in the case R = 0.
    state.set(((Nx + 1) / 2) - R - 1, "Up");
    state.set(((Nx + 1) / 2), "Dn"); // electron located at origin
    state.set(((Nx + 1) / 2) + R + 1, "Up");

    psi = IQMPS(state);
  }

  println(sweeps);

  auto E2 = psiHphi(psi, H2, psi);
  auto EV = psiHphi(psi, V, psi);

  printfln("<H2> = %.20f", E2);
  printfln("<V> = %.20f", EV);
  printfln("E = Te+Vee+Vpe = %.20f", E2 + EV);
  printfln("Etot = Te+Vee+Vpe+Vpp = %.20f", E2 + EV + Vpp);

  for (int j = 1; j <= Nx; ++j)
  {
    psi.position(j);
    Real szj = std::real((psi.A(j) * sites.op("Sz", j) * dag(prime(psi.A(j), Site))).cplx());
    Real nj = std::real((psi.A(j) * sites.op("Ntot", j) * dag(prime(psi.A(j), Site))).cplx());
    printfln("Sz_%d = %.10f   n_%d = %.10f", j, szj, j, nj);
  }

  auto Hset = std::vector<IQMPO>(2);
  Hset.at(0) = H2;
  Hset.at(1) = V;

  dmrg(psi, Hset, sweeps, "Quiet");

  println("Writing wavefunction 'psi' to disk");
  writeToFile("psi", psi);

  E2 = psiHphi(psi, H2, psi);
  EV = psiHphi(psi, V, psi);

  printfln("<H2> = %.20f", E2);
  printfln("<V> = %.20f", EV);
  printfln("E = Te+Vee+Vpe = %.20f", E2 + EV);
  printfln("Etot = Te+Vee+Vpe+Vpp = %.20f", E2 + EV + Vpp);

  double threshold = 1.0e-6;

  auto delta_E = 1.0;
  auto E_prev = E2 + EV;

  sweeps = Sweeps(2);
  auto sweep_maxm = 64;
  sweeps.maxm() = sweep_maxm;
  sweeps.cutoff() = 1E-11;
  sweeps.niter() = 4;
  sweeps.noise() = 0;
  sweeps.minm() = 20;

  auto curr_max_m = maxM(psi);

  while (delta_E > threshold)
  {
    dmrg(psi, Hset, sweeps, "Quiet");

    curr_max_m = maxM(psi);
    if (curr_max_m >= sweep_maxm - 10)
    {
      println(curr_max_m);
      sweep_maxm = sweep_maxm + 100;
      sweeps.maxm() = sweep_maxm;
    }
    else
    {
      E2 = psiHphi(psi, H2, psi);
      EV = psiHphi(psi, V, psi);

      delta_E = abs(E_prev - (E2 + EV));
      E_prev = E2 + EV;
    }
  }

  println("Writing wavefunction 'psi' to disk");
  writeToFile("psi", psi);

  E2 = psiHphi(psi, H2, psi);
  EV = psiHphi(psi, V, psi);

  printfln("<H2> = %.20f", E2);
  printfln("<V> = %.20f", EV);
  printfln("E = Te+Vee+Vpe = %.20f", E2 + EV);
  printfln("Etot = Te+Vee+Vpe+Vpp = %.20f", E2 + EV + Vpp);

  for (int j = 1; j <= Nx; ++j)
  {
    psi.position(j);
    Real szj = std::real((psi.A(j) * sites.op("Sz", j) * dag(prime(psi.A(j), Site))).cplx());
    Real nj = std::real((psi.A(j) * sites.op("Ntot", j) * dag(prime(psi.A(j), Site))).cplx());
    printfln("Sz_%d = %.20f   n_%d = %.20f", j, szj, j, nj);
  }

  return 0;
}
